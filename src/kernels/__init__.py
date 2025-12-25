"""
SPARSA-LM Custom Kernels
OpenAI Triton and CuPy acceleration
"""

from .triton_ops import (
    triton_rmsnorm,
    triton_swiglu,
    triton_rotary_embedding,
    triton_fused_attention,
    TritonRMSNorm,
    TritonSwiGLU,
)

from .cupy_ops import (
    cupy_available,
    cupy_softmax,
    cupy_layer_norm,
    cupy_gelu,
    CuPyAccelerator,
)

__all__ = [
    # Triton
    "triton_rmsnorm",
    "triton_swiglu",
    "triton_rotary_embedding",
    "triton_fused_attention",
    "TritonRMSNorm",
    "TritonSwiGLU",
    # CuPy
    "cupy_available",
    "cupy_softmax",
    "cupy_layer_norm",
    "cupy_gelu",
    "CuPyAccelerator",
]
