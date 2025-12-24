"""
SPARSA-LM High-Performance Inference Stack

Features:
- vLLM integration for high-throughput inference
- Triton custom kernels for attention acceleration
- Multi-GPU tensor parallelism (4x L4 24GB)
- KV-Cache optimization
- Continuous batching
- Speculative decoding support

This code belongs to EllanorAI and is licensed under the EllanorAI Proprietary License.
"""

import os
import sys
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F

# Optional imports with graceful fallback
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available. Install with: pip install vllm")

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logging.warning("Triton not available. Custom kernels disabled.")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

import numpy as np

# Import local model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import SPARSALM, SPARSAConfig, KVCache

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""

    # Model settings
    model_path: str = "checkpoints/sparsa-360m"
    tokenizer_path: str = "tokenizer"

    # Hardware settings
    device: str = "cuda"
    tensor_parallel_size: int = 4  # For 4x L4 GPUs
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90

    # Generation settings
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # vLLM settings
    use_vllm: bool = True
    max_num_seqs: int = 256
    max_model_len: int = 4096
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True

    # Batching settings
    max_batch_size: int = 32
    continuous_batching: bool = True

    # Speculative decoding
    use_speculative: bool = False
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5


# ============================================================================
# Triton Custom Kernels for Attention Acceleration
# ============================================================================

if TRITON_AVAILABLE:

    @triton.jit
    def _fused_attention_kernel(
        Q, K, V, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, M, N,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """Fused attention kernel with Flash Attention-style tiling."""
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        # Offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Pointers
        q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + \
                 offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        k_ptrs = K + off_z * stride_kz + off_h * stride_kh + \
                 offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        v_ptrs = V + off_z * stride_vz + off_h * stride_vh + \
                 offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

        # Load Q block
        q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

        # Scaling factor
        scale = 1.0 / tl.sqrt(tl.cast(BLOCK_K, tl.float32))

        # Initialize output accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

        # Iterate over K, V blocks
        for start_n in range(0, N, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n_curr = start_n + offs_n

            # Load K, V blocks
            k = tl.load(k_ptrs, mask=offs_n_curr[None, :] < N, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N, other=0.0)

            # Compute QK^T
            qk = tl.dot(q, k) * scale

            # Apply causal mask
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            # Online softmax
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)

            # Update accumulator
            p = tl.exp(qk - m_new[:, None])
            acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)

            # Update statistics
            m_i = m_new
            l_i = l_new

            # Advance pointers
            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn

        # Finalize output
        acc = acc / l_i[:, None]

        # Store output
        o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + \
                 offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(o_ptrs, acc, mask=offs_m[:, None] < M)

    def triton_flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Flash attention using Triton kernel."""
        batch, heads, seq_len, head_dim = q.shape
        out = torch.empty_like(q)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = head_dim

        grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

        _fused_attention_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch, heads, seq_len, seq_len,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            IS_CAUSAL=causal,
        )

        return out


# ============================================================================
# vLLM Integration
# ============================================================================

class VLLMEngine:
    """High-throughput inference engine using vLLM."""

    def __init__(self, config: InferenceConfig):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required. Install with: pip install vllm")

        self.config = config
        self.engine = None
        self.tokenizer = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the vLLM engine."""
        logging.info(f"Initializing vLLM engine with tensor_parallel_size={self.config.tensor_parallel_size}")

        # Detect available GPUs
        num_gpus = torch.cuda.device_count()
        tensor_parallel = min(self.config.tensor_parallel_size, num_gpus)

        logging.info(f"Detected {num_gpus} GPUs, using {tensor_parallel} for tensor parallelism")

        # Create LLM engine
        self.engine = LLM(
            model=self.config.model_path,
            tokenizer=self.config.tokenizer_path,
            tensor_parallel_size=tensor_parallel,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            enable_prefix_caching=self.config.enable_prefix_caching,
            trust_remote_code=True,
        )

        self.tokenizer = self.engine.get_tokenizer()
        logging.info("vLLM engine initialized successfully")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate completions for prompts."""
        if isinstance(prompts, str):
            prompts = [prompts]

        sampling_params = SamplingParams(
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            repetition_penalty=repetition_penalty or self.config.repetition_penalty,
            stop=stop,
        )

        outputs = self.engine.generate(prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Stream generation token by token."""
        # vLLM streaming implementation
        sampling_params = SamplingParams(
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
        )

        # Use async engine for streaming
        for output in self.engine.generate([prompt], sampling_params, use_tqdm=False):
            yield output.outputs[0].text


class AsyncVLLMEngine:
    """Async vLLM engine for production deployments."""

    def __init__(self, config: InferenceConfig):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required")

        self.config = config
        self.engine = None

    async def initialize(self):
        """Initialize async engine."""
        num_gpus = torch.cuda.device_count()
        tensor_parallel = min(self.config.tensor_parallel_size, num_gpus)

        engine_args = AsyncEngineArgs(
            model=self.config.model_path,
            tokenizer=self.config.tokenizer_path,
            tensor_parallel_size=tensor_parallel,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            enable_prefix_caching=self.config.enable_prefix_caching,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(
        self,
        prompt: str,
        request_id: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Async streaming generation."""
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
        )

        async for output in self.engine.generate(prompt, sampling_params, request_id):
            if output.finished:
                break
            yield output.outputs[0].text


# ============================================================================
# Native PyTorch Inference (Fallback)
# ============================================================================

class NativeInferenceEngine:
    """Native PyTorch inference engine with optimizations."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize()

    def _initialize(self):
        """Initialize model and tokenizer."""
        logging.info("Initializing native PyTorch inference engine")

        # Load model
        self.model = SPARSALM.from_pretrained(
            self.config.model_path,
            device=torch.device(self.config.device),
        )

        # Set dtype
        if self.config.dtype == "bfloat16":
            self.model = self.model.to(torch.bfloat16)
        elif self.config.dtype == "float16":
            self.model = self.model.to(torch.float16)

        self.model.eval()

        # Multi-GPU setup
        if torch.cuda.device_count() > 1:
            logging.info(f"Using {torch.cuda.device_count()} GPUs")
            # Simple data parallel (tensor parallel requires more complex setup)
            self.model = torch.nn.DataParallel(self.model)

        # Load tokenizer
        if HF_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        else:
            raise ImportError("HuggingFace transformers required for tokenizer")

        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logging.info("Model compiled with torch.compile")
            except Exception as e:
                logging.warning(f"torch.compile failed: {e}")

        logging.info("Native inference engine initialized")

    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate completions."""
        if isinstance(prompts, str):
            prompts = [prompts]

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        repetition_penalty = repetition_penalty or self.config.repetition_penalty

        results = []

        for prompt in prompts:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_model_len - max_tokens,
            ).to(self.config.device)

            # Get the base model if wrapped in DataParallel
            model = self.model.module if hasattr(self.model, 'module') else self.model

            # Generate
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Decode
            generated = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Apply stop sequences
            if stop:
                for s in stop:
                    if s in generated:
                        generated = generated[:generated.index(s)]

            results.append(generated)

        return results

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """Stream generation token by token."""
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.config.device)

        input_ids = inputs["input_ids"]
        kv_cache = KVCache()

        model = self.model.module if hasattr(self.model, 'module') else self.model

        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids[:, -1:] if kv_cache.get_seq_len() > 0 else input_ids,
                    past_key_values=kv_cache,
                    use_cache=True,
                )

            logits = outputs["logits"][:, -1, :]
            kv_cache = outputs["past_key_values"]

            # Sample next token
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Decode and yield
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text

            # Update input for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=1)


# ============================================================================
# Inference Server (OpenAI-compatible API)
# ============================================================================

class InferenceServer:
    """OpenAI-compatible inference server."""

    def __init__(self, config: InferenceConfig):
        self.config = config

        # Initialize appropriate engine
        if config.use_vllm and VLLM_AVAILABLE:
            self.engine = VLLMEngine(config)
        else:
            self.engine = NativeInferenceEngine(config)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "sparsa-360m",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """OpenAI-compatible chat completion endpoint."""
        # Format messages into prompt
        prompt = self._format_chat_messages(messages)

        if stream:
            return self._stream_response(prompt, max_tokens, temperature)
        else:
            responses = self.engine.generate(
                [prompt],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": responses[0],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(responses[0].split()),
                    "total_tokens": len(prompt.split()) + len(responses[0].split()),
                },
            }

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a single prompt."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted.append(f"<|system|>\n{content}")
            elif role == "user":
                formatted.append(f"<|user|>\n{content}")
            elif role == "assistant":
                formatted.append(f"<|assistant|>\n{content}")

        formatted.append("<|assistant|>")
        return "\n".join(formatted)

    def _stream_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ):
        """Stream response for SSE."""
        for token in self.engine.generate_stream(prompt, max_tokens, temperature):
            yield {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "sparsa-360m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }

    def completions(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """OpenAI-compatible completions endpoint."""
        responses = self.engine.generate(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "sparsa-360m",
            "choices": [
                {
                    "text": responses[0],
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }


# ============================================================================
# Utility Functions
# ============================================================================

def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware configuration."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": torch.cuda.device_count(),
        "gpu_names": [],
        "gpu_memory": [],
        "triton_available": TRITON_AVAILABLE,
        "vllm_available": VLLM_AVAILABLE,
        "cupy_available": CUPY_AVAILABLE,
    }

    if info["cuda_available"]:
        for i in range(info["num_gpus"]):
            props = torch.cuda.get_device_properties(i)
            info["gpu_names"].append(props.name)
            info["gpu_memory"].append(props.total_memory / (1024**3))  # GB

    return info


def get_optimal_config() -> InferenceConfig:
    """Get optimal inference config based on detected hardware."""
    hardware = detect_hardware()

    config = InferenceConfig()

    if hardware["cuda_available"]:
        config.device = "cuda"
        config.tensor_parallel_size = min(4, hardware["num_gpus"])

        # Adjust for GPU memory
        if hardware["gpu_memory"] and min(hardware["gpu_memory"]) < 16:
            config.max_model_len = 2048
            config.max_num_seqs = 128

        # Check for L4 GPUs
        if any("L4" in name for name in hardware["gpu_names"]):
            logging.info("Detected NVIDIA L4 GPUs - using optimized settings")
            config.tensor_parallel_size = min(4, hardware["num_gpus"])
            config.gpu_memory_utilization = 0.92
    else:
        config.device = "cpu"
        config.use_vllm = False

    config.use_vllm = hardware["vllm_available"]

    return config


# ============================================================================
# Main Entry Point
# ============================================================================

def create_inference_engine(
    model_path: str,
    tokenizer_path: str,
    **kwargs,
) -> Union[VLLMEngine, NativeInferenceEngine]:
    """Factory function to create the appropriate inference engine."""
    config = get_optimal_config()
    config.model_path = model_path
    config.tokenizer_path = tokenizer_path

    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    if config.use_vllm and VLLM_AVAILABLE:
        return VLLMEngine(config)
    else:
        return NativeInferenceEngine(config)


def main():
    """CLI interface for inference."""
    import argparse

    parser = argparse.ArgumentParser(description="SPARSA-LM Inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Show hardware info
    hardware = detect_hardware()
    print(f"Hardware: {hardware}")

    # Create engine
    engine = create_inference_engine(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
    )

    if args.serve:
        # Start API server
        try:
            from fastapi import FastAPI
            import uvicorn

            app = FastAPI(title="SPARSA-LM API")
            server = InferenceServer(get_optimal_config())

            @app.post("/v1/chat/completions")
            async def chat_completions(request: dict):
                return server.chat_completion(**request)

            @app.post("/v1/completions")
            async def completions(request: dict):
                return server.completions(**request)

            uvicorn.run(app, host="0.0.0.0", port=args.port)
        except ImportError:
            print("FastAPI/Uvicorn required for serving. Install with: pip install fastapi uvicorn")

    elif args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        while True:
            prompt = input("\nYou: ")
            if prompt.lower() == 'quit':
                break

            print("\nAssistant: ", end="", flush=True)
            for token in engine.generate_stream(prompt, args.max_tokens, args.temperature):
                print(token, end="", flush=True)
            print()

    elif args.prompt:
        responses = engine.generate(
            [args.prompt],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nGenerated: {responses[0]}")

    else:
        print("Specify --prompt, --interactive, or --serve")


if __name__ == "__main__":
    main()
