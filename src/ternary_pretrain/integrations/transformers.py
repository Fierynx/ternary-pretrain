from __future__ import annotations

from dataclasses import asdict
from typing import Any

from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from ternary_pretrain.config import ModelConfig
from ternary_pretrain.model import DecoderLM


class TernaryPretrainedConfig(PretrainedConfig):
    """Transformers serialization of the native model dimensions."""

    model_type = "ternary_pretrain"

    def __init__(
        self,
        *,
        vocab_size: int = 512,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        ffn_dim: int = 176,
        max_seq_len: int = 64,
        rope_theta: float = 10_000.0,
        rms_norm_eps: float = 1e-5,
        initializer_std: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(tie_word_embeddings=True, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_std = initializer_std

    def native(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            ffn_dim=self.ffn_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            initializer_std=self.initializer_std,
        )


class TernaryPreTrainedModel(PreTrainedModel): # type: ignore[no-untyped-call]
    """Serialization adapter that delegates logits to the native implementation."""

    config_class = TernaryPretrainedConfig # type: ignore[assignment]
    base_model_prefix = "native_model"
    _tied_weights_keys = ("native_model.lm_head.weight",) # type: ignore[assignment]

    def __init__(self, config: TernaryPretrainedConfig) -> None:
        super().__init__(config)
        self.native_model = DecoderLM(config.native())

    @classmethod
    def from_native(cls, model: DecoderLM) -> TernaryPreTrainedModel:
        config = TernaryPretrainedConfig(**asdict(model.config))
        wrapper = cls(config)
        wrapper.native_model.load_state_dict(model.state_dict(), strict=True)
        return wrapper

    def get_input_embeddings(self) -> nn.Module:
        return self.native_model.token_embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError("input embeddings must be torch.nn.Embedding")
        self.native_model.token_embedding = value
        # Keep input and output embeddings tied after replacement.
        self.native_model.lm_head.weight = value.weight

    def get_output_embeddings(self) -> nn.Module:
        return self.native_model.lm_head

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        **_: Any,
    ) -> CausalLMOutput:
        output = self.native_model(input_ids, labels=labels)
        return CausalLMOutput(loss=output.loss, logits=output.logits)
