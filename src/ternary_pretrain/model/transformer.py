from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from torch import Tensor, nn
from torch.nn import functional as F

from ternary_pretrain.config import ModelConfig
from ternary_pretrain.model.layers import DecoderBlock, RMSNorm
from ternary_pretrain.quantization import TernaryLinear


@dataclass(frozen=True, slots=True)
class ModelOutput:
    logits: Tensor
    loss: Tensor | None


class DecoderLM(nn.Module):
    """Llama-like decoder with grouped-query attention and tied embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Input and output embeddings share the same weights.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        # Scale residual output weights by model depth.
        residual_std = config.initializer_std / math.sqrt(2 * config.n_layers)
        for block in self.layers:
            block = cast(DecoderBlock, block)
            nn.init.normal_(block.attention.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.down_proj.weight, mean=0.0, std=residual_std)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, TernaryLinear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_std)

    def set_qat_enabled(self, enabled: bool) -> None:
        # Change the QAT buffers without replacing model weights.
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.set_qat_enabled(enabled)

    @property
    def qat_enabled(self) -> bool:
        values = {
            module.qat_enabled for module in self.modules() if isinstance(module, TernaryLinear)
        }
        if len(values) != 1:
            raise RuntimeError("hidden matrix QAT states are inconsistent")
        return values.pop()

    def hidden_matrix_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            module.weight for module in self.modules() if isinstance(module, TernaryLinear)
        )

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds the configured maximum")
        hidden = self.token_embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if labels is not None:
            # Compute loss in float32.
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return ModelOutput(logits=logits, loss=loss)
