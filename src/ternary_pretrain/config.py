from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_dim: int
    max_seq_len: int
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    initializer_std: float = 0.02

    def validate(self) -> None:
        dimensions = (
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.ffn_dim,
            self.max_seq_len,
        )
        if any(type(value) is not int for value in dimensions) or min(dimensions) <= 0:
            raise ConfigError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ConfigError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ConfigError("n_heads must be divisible by n_kv_heads")
        constants = (self.rope_theta, self.rms_norm_eps, self.initializer_std)
        if any(type(value) not in {int, float} for value in constants) or min(constants) <= 0:
            raise ConfigError("model floating-point constants must be positive")


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    input_files: tuple[Path, ...]
    output_file: Path
    vocab_size: int
    min_frequency: int
    eod_token: str
    max_documents: int | None = None

    def validate(self, *, require_inputs: bool = True) -> None:
        if type(self.vocab_size) is not int or not 258 <= self.vocab_size <= 65_536:
            raise ConfigError("byte-level vocab_size must be in [258, 65536]")
        if (
            type(self.min_frequency) is not int
            or self.min_frequency < 1
            or not isinstance(self.eod_token, str)
            or not self.eod_token
        ):
            raise ConfigError("min_frequency and eod_token must be non-empty positive values")
        if self.max_documents is not None and (
            type(self.max_documents) is not int or self.max_documents < 1
        ):
            raise ConfigError("tokenizer max_documents must be positive")
        if require_inputs:
            _require_files(self.input_files, "tokenizer input")


@dataclass(frozen=True, slots=True)
class DataConfig:
    mode: Literal["local", "huggingface"]
    source_files: tuple[Path, ...]
    output_dir: Path
    tokenizer_file: Path
    token_output_dir: Path
    validation_fraction: float
    seed: int
    shard_token_limit: int
    repo_id: str | None = None
    revision: str | None = None
    allow_patterns: tuple[str, ...] = ()
    max_documents: int | None = None

    def validate(self, *, operation: Literal["prepare", "tokenize"] = "prepare") -> None:
        if self.mode not in {"local", "huggingface"}:
            raise ConfigError(f"unsupported data mode: {self.mode}")
        if (
            type(self.validation_fraction) not in {int, float}
            or not 0.0 < self.validation_fraction < 1.0
        ):
            raise ConfigError("validation_fraction must be between zero and one")
        if type(self.seed) is not int:
            raise ConfigError("data seed must be an integer")
        if type(self.shard_token_limit) is not int or self.shard_token_limit < 2:
            raise ConfigError("shard_token_limit must be at least two")
        if self.mode == "local":
            if not self.source_files:
                raise ConfigError("local mode requires at least one source file")
            _require_files(self.source_files, "data source")
        elif (
            self.repo_id != "HuggingFaceFW/fineweb-edu"
            or not self.revision
            or self.revision in {"main", "master"}
            or not self.allow_patterns
            or self.max_documents is None
        ):
            raise ConfigError(
                "Hugging Face mode requires FineWeb-Edu, an immutable revision, an allow-list, "
                "and max_documents"
            )
        if self.max_documents is not None and (
            type(self.max_documents) is not int or self.max_documents < 1
        ):
            raise ConfigError("max_documents must be positive")
        if operation == "tokenize":
            _require_files(
                (self.output_dir / "train.jsonl", self.output_dir / "validation.jsonl"),
                "prepared split",
            )
            _require_files((self.tokenizer_file,), "tokenizer")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: Literal["cpu", "cuda"]
    distributed_backend: Literal["gloo", "nccl"]
    precision: Literal["float32", "bfloat16"]
    max_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    sequence_length: int
    gradient_clip_norm: float
    log_interval: int
    checkpoint_interval: int
    evaluation_interval: int
    deterministic: bool = True
    allow_tf32: bool = False

    def validate(self) -> None:
        counts = (
            self.max_steps,
            self.micro_batch_size,
            self.gradient_accumulation_steps,
            self.sequence_length,
            self.log_interval,
            self.checkpoint_interval,
            self.evaluation_interval,
        )
        if any(type(value) is not int for value in counts) or min(counts) < 1:
            raise ConfigError("runtime counts must be positive integers")
        if type(self.gradient_clip_norm) not in {int, float} or self.gradient_clip_norm <= 0:
            raise ConfigError("gradient_clip_norm must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ConfigError(f"unsupported runtime device: {self.device}")
        if self.distributed_backend not in {"gloo", "nccl"}:
            raise ConfigError(f"unsupported distributed backend: {self.distributed_backend}")
        if self.precision not in {"float32", "bfloat16"}:
            raise ConfigError(f"unsupported training precision: {self.precision}")
        if type(self.deterministic) is not bool or type(self.allow_tf32) is not bool:
            raise ConfigError("deterministic and allow_tf32 must be booleans")
        if self.device == "cpu" and self.precision != "float32":
            raise ConfigError("CPU training requires float32 precision")
        if self.distributed_backend == "nccl" and self.device != "cuda":
            raise ConfigError("NCCL requires CUDA")


OptimizerKind = Literal["adamw", "muon", "muown", "angular_muown"]


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    kind: OptimizerKind
    learning_rate: float
    auxiliary_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    momentum: float
    newton_schulz_steps: int
    angular_decay_scale: float = 0.001
    angular_decay_degree: float = 1.0

    def validate(self) -> None:
        if self.kind not in {"adamw", "muon", "muown", "angular_muown"}:
            raise ConfigError(f"unsupported optimizer: {self.kind}")
        positive_scalars = (self.learning_rate, self.auxiliary_learning_rate, self.eps)
        if (
            any(type(value) not in {int, float} for value in positive_scalars)
            or min(positive_scalars) <= 0
        ):
            raise ConfigError("learning rates and optimizer epsilon must be positive")
        if (
            type(self.weight_decay) not in {int, float}
            or self.weight_decay < 0
            or len(self.betas) != 2
            or any(type(beta) not in {int, float} for beta in self.betas)
            or not all(0 <= beta < 1 for beta in self.betas)
        ):
            raise ConfigError("weight_decay and betas are out of range")
        if (
            type(self.momentum) not in {int, float}
            or not 0 <= self.momentum < 1
            or type(self.newton_schulz_steps) is not int
            or self.newton_schulz_steps < 1
        ):
            raise ConfigError("momentum or Newton-Schulz step count is out of range")
        if (
            type(self.angular_decay_scale) not in {int, float}
            or self.angular_decay_scale <= 0
            or type(self.angular_decay_degree) not in {int, float}
            or self.angular_decay_degree < 0
        ):
            raise ConfigError("AngularMuown decay values are out of range")


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    warmup_steps: int
    stable_steps: int
    cooldown_steps: int
    final_learning_rate_ratio: float

    def validate(self, max_steps: int) -> None:
        if any(
            type(value) is not int
            for value in (self.warmup_steps, self.stable_steps, self.cooldown_steps)
        ):
            raise ConfigError("schedule phase lengths must be integers")
        if min(self.warmup_steps, self.stable_steps, self.cooldown_steps) < 0:
            raise ConfigError("schedule phases cannot be negative")
        if self.warmup_steps + self.stable_steps + self.cooldown_steps != max_steps:
            raise ConfigError("schedule phase lengths must sum to runtime.max_steps")
        if (
            type(self.final_learning_rate_ratio) not in {int, float}
            or not 0 <= self.final_learning_rate_ratio <= 1
        ):
            raise ConfigError("final_learning_rate_ratio must be in [0, 1]")


QuantizationMode = Literal["disabled", "from_init", "transition"]


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    mode: QuantizationMode
    transition_tokens: int | None = None

    def validate(self) -> None:
        if self.mode not in {"disabled", "from_init", "transition"}:
            raise ConfigError(f"unsupported quantization mode: {self.mode}")
        if self.mode == "transition" and (
            type(self.transition_tokens) is not int or self.transition_tokens < 0
        ):
            raise ConfigError("transition mode requires a non-negative transition_tokens")
        if self.mode != "transition" and self.transition_tokens is not None:
            raise ConfigError("transition_tokens is only valid for transition mode")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    tensorboard: bool
    wandb: bool
    wandb_project: str | None = None

    def validate(self) -> None:
        if type(self.tensorboard) is not bool or type(self.wandb) is not bool:
            raise ConfigError("tracking switches must be booleans")
        if not self.tensorboard:
            raise ConfigError("TensorBoard tracking is always enabled")
        if self.wandb and not self.wandb_project:
            raise ConfigError("wandb_project is required when W&B tracking is enabled")


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    output_dir: Path
    model_config: Path
    train_manifest: Path
    validation_manifest: Path
    tokenizer_file: Path
    seed: int
    runtime: RuntimeConfig
    optimizer: OptimizerConfig
    schedule: ScheduleConfig
    quantization: QuantizationConfig
    tracking: TrackingConfig

    def validate(self, *, require_artifacts: bool = True) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or any(character in self.name for character in "\\/:")
        ):
            raise ConfigError("run name is empty or contains a path separator")
        if type(self.seed) is not int:
            raise ConfigError("run seed must be an integer")
        self.runtime.validate()
        self.optimizer.validate()
        self.schedule.validate(self.runtime.max_steps)
        self.quantization.validate()
        self.tracking.validate()
        _require_files((self.model_config,), "model config")
        if require_artifacts:
            _require_files(
                (self.train_manifest, self.validation_manifest, self.tokenizer_file),
                "run artifact",
            )


def _require_files(paths: tuple[Path, ...], label: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ConfigError(f"missing {label} file(s): {', '.join(missing)}")


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {path}: {error}") from error


def _resolve(config_dir: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        (config_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    )


def _decode_dataclass[T](data: dict[str, Any], cls: type[T], *, context: str) -> T:
    # Reject typos instead of ignoring unknown settings.
    fields = dataclasses.fields(cast(Any, cls))
    allowed = {field.name for field in fields}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) in {context}: {', '.join(sorted(unknown))}")
    missing = {
        field.name
        for field in fields
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
        and field.name not in data
    }
    if missing:
        raise ConfigError(f"missing key(s) in {context}: {', '.join(sorted(missing))}")
    try:
        return cls(**data)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid value in {context}: {error}") from error


def load_model_config(path: str | Path) -> ModelConfig:
    config = _decode_dataclass(
        _read_toml(Path(path).resolve()), ModelConfig, context="model config"
    )
    config.validate()
    return config


def load_tokenizer_config(path: str | Path, *, require_inputs: bool = True) -> TokenizerConfig:
    resolved = Path(path).resolve()
    document = _read_toml(resolved)
    config_dir = resolved.parent
    document["input_files"] = tuple(
        _resolve(config_dir, value) for value in document.get("input_files", ())
    )
    if "output_file" in document:
        document["output_file"] = _resolve(config_dir, document["output_file"])
    config = _decode_dataclass(document, TokenizerConfig, context="tokenizer config")
    config.validate(require_inputs=require_inputs)
    return config


def load_data_config(
    path: str | Path, *, operation: Literal["prepare", "tokenize"] = "prepare"
) -> DataConfig:
    resolved = Path(path).resolve()
    document = _read_toml(resolved)
    config_dir = resolved.parent
    document["source_files"] = tuple(
        _resolve(config_dir, value) for value in document.get("source_files", ())
    )
    document["allow_patterns"] = tuple(document.get("allow_patterns", ()))
    for key in ("output_dir", "tokenizer_file", "token_output_dir"):
        if key in document:
            document[key] = _resolve(config_dir, document[key])
    config = _decode_dataclass(document, DataConfig, context="data config")
    config.validate(operation=operation)
    return config


def load_run_config(path: str | Path, *, require_artifacts: bool = True) -> RunConfig:
    resolved = Path(path).resolve()
    document = _read_toml(resolved)
    config_dir = resolved.parent
    sections = {"runtime", "optimizer", "schedule", "quantization", "tracking"}
    expected_root = {
        "name",
        "output_dir",
        "model_config",
        "train_manifest",
        "validation_manifest",
        "tokenizer_file",
        "seed",
        *sections,
    }
    unknown = set(document) - expected_root
    if unknown:
        raise ConfigError(f"unknown key(s) in run config: {', '.join(sorted(unknown))}")
    missing_sections = sections - set(document)
    if missing_sections:
        raise ConfigError(f"missing run section(s): {', '.join(sorted(missing_sections))}")
    for key in (
        "output_dir",
        "model_config",
        "train_manifest",
        "validation_manifest",
        "tokenizer_file",
    ):
        if key in document:
            document[key] = _resolve(config_dir, document[key])
    optimizer_data = dict(document.pop("optimizer"))
    optimizer_data["betas"] = tuple(optimizer_data.get("betas", ()))
    config = RunConfig(
        runtime=_decode_dataclass(document.pop("runtime"), RuntimeConfig, context="runtime"),
        optimizer=_decode_dataclass(optimizer_data, OptimizerConfig, context="optimizer"),
        schedule=_decode_dataclass(document.pop("schedule"), ScheduleConfig, context="schedule"),
        quantization=_decode_dataclass(
            document.pop("quantization"), QuantizationConfig, context="quantization"
        ),
        tracking=_decode_dataclass(document.pop("tracking"), TrackingConfig, context="tracking"),
        **document,
    )
    config.validate(require_artifacts=require_artifacts)
    return config


def canonical_data(value: object) -> object:
    # Normalize values before hashing them.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: canonical_data(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [canonical_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonical_data(item) for key, item in sorted(value.items())}
    return cast(str | int | float | bool | None, value)


def config_hash(value: object) -> str:
    payload = json.dumps(canonical_data(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
