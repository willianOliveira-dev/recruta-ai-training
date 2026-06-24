from __future__ import annotations

import logging

from unsloth import FastLanguageModel

from src.config.settings import AppSettings
from src.shared.protocols import ChatTemplateTokenizer

logger = logging.getLogger(__name__)


class UnslothModelLoader:
    def __init__(self, config: AppSettings) -> None:
        self._config = config

    def load_base_model(self) -> tuple[object, ChatTemplateTokenizer]:
        logger.info("Loading model: %s", self._config.model.name_or_path)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self._config.model.name_or_path,
            max_seq_length=self._config.training.max_seq_length,
            dtype=None,
            load_in_4bit=self._config.model.load_in_4bit,
        )
        return model, tokenizer

    def setup_peft(self, model: object) -> object:
        logger.info("Applying PEFT/LoRA adapters (r=%d, alpha=%d)", self._config.peft.r, self._config.peft.lora_alpha)
        return FastLanguageModel.get_peft_model(
            model,
            r=self._config.peft.r,
            target_modules=self._config.peft.target_modules,
            lora_alpha=self._config.peft.lora_alpha,
            lora_dropout=self._config.peft.lora_dropout,
            bias=self._config.peft.bias,
            use_gradient_checkpointing=self._config.peft.use_gradient_checkpointing,
            random_state=self._config.seed,
            use_rslora=self._config.peft.use_rslora,
            loftq_config=None,
        )
