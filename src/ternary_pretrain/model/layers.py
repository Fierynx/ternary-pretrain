from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ternary_pretrain.config import ModelConfig
from ternary_pretrain.quantization import TernaryLinear


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        # Compute the norm in float32 for stability.
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(inputs.dtype)


def rope_frequencies(
    sequence_length: int, head_dim: int, theta: float, *, device: torch.device
) -> tuple[Tensor, Tensor]:
    inverse = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse)
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def apply_rope(inputs: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    even, odd = inputs[..., 0::2], inputs[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2).to(inputs.dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.rope_theta = config.rope_theta
        self.q_proj = TernaryLinear(config.d_model, config.n_heads * self.head_dim)
        self.k_proj = TernaryLinear(config.d_model, config.n_kv_heads * self.head_dim)
        self.v_proj = TernaryLinear(config.d_model, config.n_kv_heads * self.head_dim)
        self.o_proj = TernaryLinear(config.n_heads * self.head_dim, config.d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, _ = inputs.shape
        query = (
            self.q_proj(inputs)
            .view(batch_size, sequence_length, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(inputs)
            .view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(inputs)
            .view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        cosine, sine = rope_frequencies(
            sequence_length, self.head_dim, self.rope_theta, device=inputs.device
        )
        query = apply_rope(query, cosine, sine)
        key = apply_rope(key, cosine, sine)
        # Expand shared K/V heads to match the query heads used by SDPA.
        queries_per_kv_head = self.n_heads // self.n_kv_heads
        key = key.repeat_interleave(queries_per_kv_head, dim=1)
        value = value.repeat_interleave(queries_per_kv_head, dim=1)
        attention_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention_output = (
            attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        )
        return cast(Tensor, self.o_proj(attention_output))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = TernaryLinear(config.d_model, config.ffn_dim)
        self.up_proj = TernaryLinear(config.d_model, config.ffn_dim)
        self.down_proj = TernaryLinear(config.ffn_dim, config.d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs)))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return cast(Tensor, hidden + self.feed_forward(self.ffn_norm(hidden)))
