from __future__ import annotations

import logging

from datasets import Dataset

from src.shared.protocols import ChatTemplateTokenizer

logger = logging.getLogger(__name__)


class DatasetPreparator:
    def __init__(self, tokenizer: ChatTemplateTokenizer) -> None:
        self._tokenizer = tokenizer

    def _format_chat_template(self, example: dict[str, str]) -> dict[str, str]:
        messages = [
            {"role": "user", "content": example.get("instruction", "")},
            {"role": "assistant", "content": example.get("output", "")},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    def _format_generic(self, example: dict) -> dict[str, str]:
        text_parts = []
        for k, v in example.items():
            if v is None:
                continue
            str_value = str(v) if not isinstance(v, str) else v
            if str_value.strip():
                text_parts.append(f"{k}: {str_value}")
        return {"text": "\n".join(text_parts)}

    def _filter_empty_text(self, example: dict[str, str]) -> bool:
        return bool(example.get("text", "").strip())

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        has_instruction = "instruction" in dataset.column_names
        has_output = "output" in dataset.column_names

        if has_instruction and has_output:
            logger.info("Formatting dataset with chat template")
            formatted = dataset.map(self._format_chat_template, num_proc=4)
        elif "text" in dataset.column_names:
            logger.info("Dataset already has 'text' column")
            formatted = dataset
        else:
            logger.warning(
                "Dataset missing 'instruction'/'output' columns. Using generic formatting."
            )
            formatted = dataset.map(self._format_generic, num_proc=4)

        before_count = len(formatted)
        formatted = formatted.filter(self._filter_empty_text, num_proc=4)
        after_count = len(formatted)

        if before_count != after_count:
            logger.info(
                "Filtered %d empty rows (before=%d, after=%d)",
                before_count - after_count,
                before_count,
                after_count,
            )

        return formatted

