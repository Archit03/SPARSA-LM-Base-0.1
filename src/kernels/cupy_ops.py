"""
SPARSA-LM CuPy Accelerated Operations
High-performance GPU operations using CuPy with custom CUDA kernels
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math

# Check if CuPy is available
try:
    import cupy as cp
    from cupy import cuda
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def cupy_available() -> bool:
    """Check if CuPy is available and CUDA is accessible."""
    if not CUPY_AVAILABLE:
        return False
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


# =============================================================================
# CUDA KERNEL STRINGS
# =============================================================================

SOFTMAX_KERNEL = r'''
extern "C" __global__
void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N,
    const int D
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_val = shared + blockDim.x;

    // Find max value in row (for numerical stability)
    float thread_max = -INFINITY;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = input[row * D + i];
        thread_max = fmaxf(thread_max, val);
    }
    max_val[tid] = thread_max;
    __syncthreads();

    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            max_val[tid] = fmaxf(max_val[tid], max_val[tid + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = expf(input[row * D + i] - row_max);
        output[row * D + i] = val;
        thread_sum += val;
    }
    sum_val[tid] = thread_sum;
    __syncthreads();

    // Reduce to find global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_val[tid] += sum_val[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sum_val[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < D; i += blockDim.x) {
        output[row * D + i] /= row_sum;
    }
}
'''

LAYER_NORM_KERNEL = r'''
extern "C" __global__
void layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    const int N,
    const int D,
    const float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    // Compute mean
    float thread_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        thread_sum += input[row * D + i];
    }
    mean_shared[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mean_shared[tid] += mean_shared[tid + s];
        }
        __syncthreads();
    }
    float mean = mean_shared[0] / D;
    __syncthreads();

    // Compute variance
    float thread_var = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float diff = input[row * D + i] - mean;
        thread_var += diff * diff;
    }
    var_shared[tid] = thread_var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            var_shared[tid] += var_shared[tid + s];
        }
        __syncthreads();
    }
    float var = var_shared[0] / D;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    // Normalize and scale
    for (int i = tid; i < D; i += blockDim.x) {
        float normalized = (input[row * D + i] - mean) * inv_std;
        output[row * D + i] = gamma[i] * normalized + beta[i];
    }
}
'''

RMS_NORM_KERNEL = r'''
extern "C" __global__
void rms_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int N,
    const int D,
    const float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    // Compute sum of squares
    float thread_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = input[row * D + i];
        thread_sum += val * val;
    }
    shared[tid] = thread_sum;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / D + eps);
    __syncthreads();

    // Normalize and scale
    for (int i = tid; i < D; i += blockDim.x) {
        output[row * D + i] = input[row * D + i] * rms * weight[i];
    }
}
'''

GELU_KERNEL = r'''
extern "C" __global__
void gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}
'''

SILU_KERNEL = r'''
extern "C" __global__
void silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        // SiLU(x) = x * sigmoid(x)
        output[idx] = x / (1.0f + expf(-x));
    }
}
'''

SWIGLU_KERNEL = r'''
extern "C" __global__
void swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float g = gate[idx];
        float u = up[idx];
        // SwiGLU(gate, up) = SiLU(gate) * up = (gate * sigmoid(gate)) * up
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = silu_g * u;
    }
}
'''

ROPE_KERNEL = r'''
extern "C" __global__
void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;
    int tid = threadIdx.x;

    int half_dim = head_dim / 2;

    for (int d = tid; d < half_dim; d += blockDim.x) {
        int base_idx = b * num_heads * seq_len * head_dim +
                       h * seq_len * head_dim +
                       s * head_dim;

        float cos_val = cos_cache[s * half_dim + d];
        float sin_val = sin_cache[s * half_dim + d];

        // Apply rotation to Q
        float q1 = q[base_idx + d];
        float q2 = q[base_idx + d + half_dim];
        q[base_idx + d] = q1 * cos_val - q2 * sin_val;
        q[base_idx + d + half_dim] = q1 * sin_val + q2 * cos_val;

        // Apply rotation to K
        float k1 = k[base_idx + d];
        float k2 = k[base_idx + d + half_dim];
        k[base_idx + d] = k1 * cos_val - k2 * sin_val;
        k[base_idx + d + half_dim] = k1 * sin_val + k2 * cos_val;
    }
}
'''

FUSED_ADD_NORM_KERNEL = r'''
extern "C" __global__
void fused_add_norm_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ hidden,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ residual_out,
    const int N,
    const int D,
    const float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    // First pass: add residual and compute sum of squares
    float thread_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = residual[row * D + i] + hidden[row * D + i];
        residual_out[row * D + i] = val;  // Store updated residual
        thread_sum += val * val;
    }
    shared[tid] = thread_sum;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / D + eps);
    __syncthreads();

    // Second pass: normalize and scale
    for (int i = tid; i < D; i += blockDim.x) {
        output[row * D + i] = residual_out[row * D + i] * rms * weight[i];
    }
}
'''


# =============================================================================
# CUPY KERNEL COMPILATION
# =============================================================================

class CuPyKernelCache:
    """Cache for compiled CuPy kernels."""

    _instance = None
    _kernels = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_kernel(self, name: str, code: str) -> Optional['cp.RawKernel']:
        """Get or compile a kernel."""
        if not CUPY_AVAILABLE:
            return None

        if name not in self._kernels:
            try:
                self._kernels[name] = cp.RawKernel(code, name)
            except Exception as e:
                print(f"Failed to compile kernel {name}: {e}")
                return None
        return self._kernels[name]


# =============================================================================
# CUPY OPERATIONS
# =============================================================================

def cupy_softmax(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Softmax using CuPy CUDA kernel.

    Args:
        x: Input tensor
        dim: Dimension to apply softmax over

    Returns:
        Softmax output
    """
    if not CUPY_AVAILABLE or not x.is_cuda:
        return torch.softmax(x, dim=dim)

    cache = CuPyKernelCache()
    kernel = cache.get_kernel('softmax_kernel', SOFTMAX_KERNEL)

    if kernel is None:
        return torch.softmax(x, dim=dim)

    # Reshape for kernel
    original_shape = x.shape
    if dim != -1 and dim != len(original_shape) - 1:
        x = x.transpose(dim, -1).contiguous()

    x = x.view(-1, x.shape[-1]).contiguous()
    N, D = x.shape

    # Allocate output
    output = torch.empty_like(x)

    # Convert to CuPy arrays
    x_cp = cp.asarray(x.detach())
    output_cp = cp.asarray(output)

    # Launch kernel
    block_size = min(256, D)
    shared_mem = 2 * block_size * 4  # 2 arrays of floats

    kernel(
        (N,), (block_size,),
        (x_cp, output_cp, N, D),
        shared_mem=shared_mem,
    )

    # Convert back
    output = torch.as_tensor(output_cp, device=x.device)
    output = output.view(*original_shape[:-1], -1)

    if dim != -1 and dim != len(original_shape) - 1:
        output = output.transpose(dim, -1).contiguous()

    return output


def cupy_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Layer normalization using CuPy CUDA kernel.

    Args:
        x: Input tensor (batch, seq_len, hidden_size)
        weight: Gamma parameter (hidden_size,)
        bias: Beta parameter (hidden_size,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    if not CUPY_AVAILABLE or not x.is_cuda:
        return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)

    cache = CuPyKernelCache()
    kernel = cache.get_kernel('layer_norm_kernel', LAYER_NORM_KERNEL)

    if kernel is None:
        return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)

    # Reshape
    original_shape = x.shape
    x = x.view(-1, x.shape[-1]).contiguous()
    N, D = x.shape

    # Allocate output
    output = torch.empty_like(x)

    # Convert to CuPy
    x_cp = cp.asarray(x.detach())
    weight_cp = cp.asarray(weight.detach())
    bias_cp = cp.asarray(bias.detach())
    output_cp = cp.asarray(output)

    # Launch kernel
    block_size = min(256, D)
    shared_mem = 2 * block_size * 4

    kernel(
        (N,), (block_size,),
        (x_cp, weight_cp, bias_cp, output_cp, N, D, eps),
        shared_mem=shared_mem,
    )

    output = torch.as_tensor(output_cp, device=x.device)
    return output.view(original_shape)


def cupy_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    RMS normalization using CuPy CUDA kernel.

    Args:
        x: Input tensor
        weight: Weight parameter
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    if not CUPY_AVAILABLE or not x.is_cuda:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return x_normed * weight

    cache = CuPyKernelCache()
    kernel = cache.get_kernel('rms_norm_kernel', RMS_NORM_KERNEL)

    if kernel is None:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return x_normed * weight

    original_shape = x.shape
    x = x.view(-1, x.shape[-1]).contiguous()
    N, D = x.shape

    output = torch.empty_like(x)

    x_cp = cp.asarray(x.detach())
    weight_cp = cp.asarray(weight.detach())
    output_cp = cp.asarray(output)

    block_size = min(256, D)
    shared_mem = block_size * 4

    kernel(
        (N,), (block_size,),
        (x_cp, weight_cp, output_cp, N, D, eps),
        shared_mem=shared_mem,
    )

    output = torch.as_tensor(output_cp, device=x.device)
    return output.view(original_shape)


def cupy_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation using CuPy CUDA kernel.

    Args:
        x: Input tensor

    Returns:
        GELU output
    """
    if not CUPY_AVAILABLE or not x.is_cuda:
        return torch.nn.functional.gelu(x)

    cache = CuPyKernelCache()
    kernel = cache.get_kernel('gelu_kernel', GELU_KERNEL)

    if kernel is None:
        return torch.nn.functional.gelu(x)

    N = x.numel()
    output = torch.empty_like(x)

    x_cp = cp.asarray(x.view(-1).detach())
    output_cp = cp.asarray(output.view(-1))

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (x_cp, output_cp, N),
    )

    output = torch.as_tensor(output_cp, device=x.device)
    return output.view(x.shape)


def cupy_silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Swish) activation using CuPy CUDA kernel.

    Args:
        x: Input tensor

    Returns:
        SiLU output
    """
    if not CUPY_AVAILABLE or not x.is_cuda:
        return torch.nn.functional.silu(x)

    cache = CuPyKernelCache()
    kernel = cache.get_kernel('silu_kernel', SILU_KERNEL)

    if kernel is None:
        return torch.nn.functional.silu(x)

    N = x.numel()
    output = torch.empty_like(x)

    x_cp = cp.asarray(x.view(-1).detach())
    output_cp = cp.asarray(output.view(-1))

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (x_cp, output_cp, N),
    )

    output = torch.as_tensor(output_cp, device=x.device)
    return output.view(x.shape)


def cupy_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    SwiGLU activation using CuPy CUDA kernel.

    Args:
        gate: Gate projection output
        up: Up projection output

    Returns:
        SwiGLU output: SiLU(gate) * up
    """
    if not CUPY_AVAILABLE or not gate.is_cuda:
        return torch.nn.functional.silu(gate) * up

    cache = CuPyKernelCache()
    kernel = cache.get_kernel('swiglu_kernel', SWIGLU_KERNEL)

    if kernel is None:
        return torch.nn.functional.silu(gate) * up

    assert gate.shape == up.shape
    N = gate.numel()
    output = torch.empty_like(gate)

    gate_cp = cp.asarray(gate.view(-1).detach())
    up_cp = cp.asarray(up.view(-1).detach())
    output_cp = cp.asarray(output.view(-1))

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (gate_cp, up_cp, output_cp, N),
    )

    output = torch.as_tensor(output_cp, device=gate.device)
    return output.view(gate.shape)


def cupy_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary position embeddings using CuPy CUDA kernel.

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        cos: Cosine embeddings
        sin: Sine embeddings

    Returns:
        Tuple of rotated (q, k)
    """
    if not CUPY_AVAILABLE or not q.is_cuda:
        # Fallback to PyTorch
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

        return q_rot, k_rot

    cache = CuPyKernelCache()
    kernel = cache.get_kernel('rope_kernel', ROPE_KERNEL)

    if kernel is None:
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        return q_rot, k_rot

    batch_size, num_heads, seq_len, head_dim = q.shape

    q_out = q.clone().contiguous()
    k_out = k.clone().contiguous()

    q_cp = cp.asarray(q_out)
    k_cp = cp.asarray(k_out)
    cos_cp = cp.asarray(cos.contiguous())
    sin_cp = cp.asarray(sin.contiguous())

    block_size = min(128, head_dim // 2)

    kernel(
        (batch_size, num_heads, seq_len), (block_size,),
        (q_cp, k_cp, cos_cp, sin_cp, batch_size, num_heads, seq_len, head_dim),
    )

    q_out = torch.as_tensor(q_cp, device=q.device)
    k_out = torch.as_tensor(k_cp, device=k.device)

    return q_out, k_out


# =============================================================================
# TORCH.NN MODULES WITH CUPY BACKEND
# =============================================================================

class CuPyRMSNorm(nn.Module):
    """RMS Normalization with CuPy acceleration."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cupy_rms_norm(x, self.weight, self.eps)


class CuPyLayerNorm(nn.Module):
    """Layer Normalization with CuPy acceleration."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cupy_layer_norm(x, self.weight, self.bias, self.eps)


class CuPySwiGLU(nn.Module):
    """SwiGLU MLP with CuPy acceleration."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = cupy_swiglu(gate, up)
        return self.down_proj(activated)


# =============================================================================
# CUPY ACCELERATOR CLASS
# =============================================================================

class CuPyAccelerator:
    """
    Accelerator class for managing CuPy operations.

    Provides unified interface for GPU acceleration with automatic fallback.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize CuPy accelerator.

        Args:
            device: Target CUDA device
        """
        self.available = cupy_available()
        self.device = device or (torch.device('cuda:0') if torch.cuda.is_available() else None)
        self._kernel_cache = CuPyKernelCache() if self.available else None

        if self.available:
            self._setup_memory_pool()

    def _setup_memory_pool(self):
        """Setup CuPy memory pool for efficient allocation."""
        if not self.available:
            return

        # Use default memory pool with automatic allocation
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(fraction=0.8)  # Use up to 80% of GPU memory

    def softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Accelerated softmax."""
        return cupy_softmax(x, dim)

    def layer_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Accelerated layer normalization."""
        return cupy_layer_norm(x, weight, bias, eps)

    def rms_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Accelerated RMS normalization."""
        return cupy_rms_norm(x, weight, eps)

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        """Accelerated GELU activation."""
        return cupy_gelu(x)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """Accelerated SiLU activation."""
        return cupy_silu(x)

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Accelerated SwiGLU activation."""
        return cupy_swiglu(gate, up)

    def rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Accelerated rotary position embeddings."""
        return cupy_rope(q, k, cos, sin)

    def synchronize(self):
        """Synchronize CUDA stream."""
        if self.available and self.device is not None:
            cp.cuda.Stream.null.synchronize()

    def clear_cache(self):
        """Clear CuPy memory pool cache."""
        if self.available:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

    @property
    def memory_info(self) -> dict:
        """Get GPU memory information."""
        if not self.available:
            return {"available": False}

        mempool = cp.get_default_memory_pool()
        return {
            "available": True,
            "used_bytes": mempool.used_bytes(),
            "total_bytes": mempool.total_bytes(),
            "n_free_blocks": mempool.n_free_blocks(),
        }

    def benchmark(self, func, *args, warmup: int = 5, runs: int = 20, **kwargs) -> dict:
        """
        Benchmark a function.

        Args:
            func: Function to benchmark
            *args: Function arguments
            warmup: Number of warmup runs
            runs: Number of timed runs
            **kwargs: Function keyword arguments

        Returns:
            Dict with timing statistics
        """
        import time

        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)

        if self.available:
            self.synchronize()

        # Timed runs
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            if self.available:
                self.synchronize()
            times.append(time.perf_counter() - start)

        import statistics
        return {
            "mean_ms": statistics.mean(times) * 1000,
            "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_cupy_device_info() -> dict:
    """Get CuPy device information."""
    if not CUPY_AVAILABLE:
        return {"available": False, "error": "CuPy not installed"}

    try:
        device = cp.cuda.Device(0)
        return {
            "available": True,
            "name": device.name,
            "compute_capability": device.compute_capability,
            "total_memory_gb": device.mem_info[1] / (1024**3),
            "free_memory_gb": device.mem_info[0] / (1024**3),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def run_cupy_tests() -> dict:
    """Run tests to verify CuPy operations."""
    results = {}

    if not CUPY_AVAILABLE:
        return {"error": "CuPy not available"}

    try:
        device = torch.device('cuda:0')

        # Test softmax
        x = torch.randn(4, 8, 256, device=device)
        out_cupy = cupy_softmax(x, dim=-1)
        out_torch = torch.softmax(x, dim=-1)
        results["softmax"] = {
            "passed": torch.allclose(out_cupy, out_torch, atol=1e-5),
            "max_diff": (out_cupy - out_torch).abs().max().item(),
        }

        # Test GELU
        x = torch.randn(4, 8, 256, device=device)
        out_cupy = cupy_gelu(x)
        out_torch = torch.nn.functional.gelu(x)
        results["gelu"] = {
            "passed": torch.allclose(out_cupy, out_torch, atol=1e-4),
            "max_diff": (out_cupy - out_torch).abs().max().item(),
        }

        # Test SiLU
        x = torch.randn(4, 8, 256, device=device)
        out_cupy = cupy_silu(x)
        out_torch = torch.nn.functional.silu(x)
        results["silu"] = {
            "passed": torch.allclose(out_cupy, out_torch, atol=1e-5),
            "max_diff": (out_cupy - out_torch).abs().max().item(),
        }

        # Test SwiGLU
        gate = torch.randn(4, 8, 256, device=device)
        up = torch.randn(4, 8, 256, device=device)
        out_cupy = cupy_swiglu(gate, up)
        out_torch = torch.nn.functional.silu(gate) * up
        results["swiglu"] = {
            "passed": torch.allclose(out_cupy, out_torch, atol=1e-5),
            "max_diff": (out_cupy - out_torch).abs().max().item(),
        }

        # Test RMS Norm
        x = torch.randn(4, 8, 256, device=device)
        weight = torch.ones(256, device=device)
        out_cupy = cupy_rms_norm(x, weight, eps=1e-6)
        variance = x.pow(2).mean(-1, keepdim=True)
        out_torch = x * torch.rsqrt(variance + 1e-6) * weight
        results["rms_norm"] = {
            "passed": torch.allclose(out_cupy, out_torch, atol=1e-4),
            "max_diff": (out_cupy - out_torch).abs().max().item(),
        }

        results["all_passed"] = all(r.get("passed", False) for r in results.values() if isinstance(r, dict))

    except Exception as e:
        results["error"] = str(e)

    return results
