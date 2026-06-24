from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

from src.config.settings import AppSettings
from src.shared.errors import TrainingError
from src.shared.protocols import ChatTemplateTokenizer

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        config: AppSettings,
        model: object,
        tokenizer: ChatTemplateTokenizer,
        train_dataset: Dataset,
    ) -> None:
        self._config = config
        self._model = model
        self._tokenizer = tokenizer
        self._train_dataset = train_dataset

    def _build_training_arguments(self) -> TrainingArguments:
        return TrainingArguments(
            per_device_train_batch_size=self._config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self._config.training.gradient_accumulation_steps,
            warmup_steps=self._config.training.warmup_steps,
            warmup_ratio=self._config.training.warmup_ratio,
            max_steps=self._config.training.max_steps,
            num_train_epochs=self._config.training.num_train_epochs,
            learning_rate=self._config.training.learning_rate,
            fp16=not is_bfloat16_supported() and self._config.training.fp16,
            bf16=is_bfloat16_supported() and self._config.training.bf16,
            logging_steps=self._config.training.logging_steps,
            optim=self._config.training.optim,
            weight_decay=self._config.training.weight_decay,
            lr_scheduler_type=self._config.training.lr_scheduler_type,
            seed=self._config.seed,
            output_dir=self._config.output_dir,
            save_steps=self._config.training.save_steps,
            report_to="wandb" if self._config.wandb.enabled else "none",
            run_name=self._config.wandb.name if self._config.wandb.enabled else None,
        )

    def train(self) -> object:
        logger.info("Initializing SFTTrainer")
        try:
            trainer = SFTTrainer(
                model=self._model,
                tokenizer=self._tokenizer,
                train_dataset=self._train_dataset,
                dataset_text_field="text",
                max_seq_length=self._config.training.max_seq_length,
                dataset_num_proc=2,
                packing=self._config.training.packing,
                args=self._build_training_arguments(),
            )
            logger.info("Starting training")
            trainer_stats = trainer.train()
            logger.info("Training completed")
            return trainer_stats
        except Exception as exc:
            raise TrainingError(str(exc)) from exc

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(output_dir))
        self._tokenizer.save_pretrained(str(output_dir))
        logger.info("Model saved to: %s", output_dir)
