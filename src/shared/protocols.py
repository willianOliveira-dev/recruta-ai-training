from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from datasets import Dataset

    from src.config.settings import AppSettings, DataSourceConfig


@runtime_checkable
class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str: ...


@runtime_checkable
class ModelLoader(Protocol):
    def load_base_model(self) -> tuple[object, ChatTemplateTokenizer]: ...
    def setup_peft(self, model: object) -> object: ...


@runtime_checkable
class DataSourceStrategy(Protocol):
    def ingest(self, source: DataSourceConfig, output_dir: Path) -> Dataset | Path: ...


@runtime_checkable
class Trainer(Protocol):
    def train(self) -> object: ...
    def save(self, output_dir: Path) -> None: ...
