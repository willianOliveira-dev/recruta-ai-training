from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.shared.errors import ConfigurationLoadError

logger = logging.getLogger(__name__)

DataSourceType = Literal["huggingface", "kaggle", "local", "s3", "gcs"]


class ModelConfig(BaseModel):
    name_or_path: str
    load_in_4bit: bool = True


class TrainingConfig(BaseModel):
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    max_steps: int = -1
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 10
    warmup_ratio: float = 0.03
    logging_steps: int = 5
    save_steps: int = 50
    fp16: bool = False
    bf16: bool = True
    packing: bool = False


class PeftConfig(BaseModel):
    r: int = 16
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str | bool = "unsloth"
    use_rslora: bool = False

    @model_validator(mode="after")
    def validate_alpha_rank_ratio(self) -> PeftConfig:
        if self.lora_alpha < self.r:
            logger.warning(
                "lora_alpha (%d) < r (%d). Recommended: lora_alpha >= 2 * r",
                self.lora_alpha,
                self.r,
            )
        return self


class DataSourceConfig(BaseModel):
    name: str
    source_type: DataSourceType
    uri: str


class DataConfig(BaseModel):
    sources: list[DataSourceConfig] = Field(default_factory=list)


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "recruta-ai-training"
    name: str | None = None


class PublishConfig(BaseModel):
    enabled: bool = False
    repo_id: str = "seu-usuario/recruta-ats-micro-3b-gguf"
    merge_before_push: bool = False
    push_gguf: bool = True
    gguf_quantization: str = "q4_k_m"
    token_env_var: str = "HF_TOKEN"


class AppSettings(BaseModel):
    project_name: str = "recruta-ai"
    seed: int = 42
    output_dir: str = "./outputs"
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    peft: PeftConfig = Field(default_factory=PeftConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)

    @classmethod
    def from_yaml(
        cls,
        base_path: str = "configs/base.yaml",
        profile_path: str = "configs/lite.yaml",
    ) -> AppSettings:
        config_dict = _load_and_merge_yaml(
            Path(base_path),
            Path(profile_path),
        )
        return cls(**config_dict)


def _load_yaml_file(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise ConfigurationLoadError(path, "File not found") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationLoadError(path, str(exc)) from exc


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_and_merge_yaml(base_path: Path, profile_path: Path) -> dict:
    base_data = _load_yaml_file(base_path)
    profile_data = _load_yaml_file(profile_path)
    merged = _deep_merge(base_data, profile_data)

    required_sections = ("model", "training", "peft", "data", "wandb", "publish")
    for section in required_sections:
        merged.setdefault(section, {})

    return merged
