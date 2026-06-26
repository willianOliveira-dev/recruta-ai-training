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
        # Fallback genérico: junta todos os valores da linha em um texto
        text_parts = []
        for k, v in example.items():
            if v and isinstance(v, str):
                text_parts.append(f"{k}: {v}")
        return {"text": "\n".join(text_parts)}

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        has_instruction = "instruction" in dataset.column_names
        has_output = "output" in dataset.column_names

        if has_instruction and has_output:
            logger.info("Formatting dataset with chat template")
            return dataset.map(self._format_chat_template, num_proc=4)

        logger.warning(
            "Dataset missing 'instruction'/'output' columns. Using generic formatting."
        )
        if "text" not in dataset.column_names:
            return dataset.map(self._format_generic, num_proc=4)
        return dataset
