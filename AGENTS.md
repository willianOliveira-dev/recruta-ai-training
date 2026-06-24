# Recruta AI Training Agent Guidelines

## Mandatory Project Skills

For every implementation, refactor, review, architecture decision, test, or debugging task in this repository, use these project skills before making changes:

- `.agents/skills/python-expert`
- `.agents/skills/python-type-safety`
- `.agents/skills/solid`
- `.agents/skills/fine-tuning-expert`
- `.agents/skills/llm-fine-tuning-guide`
- `.agents/skills/peft-fine-tuning`

The Python, type-safety, and SOLID skills define the engineering baseline. The fine-tuning, PEFT, and LLM guide skills define the AI training intelligence baseline.

These fine-tuning skills are not generic data-science instructions. Use them as adversarial and domain context for building a stronger, more efficient training pipeline: understand how to optimize LoRA ranks, configure 4-bit quantization, structure prompt templates, and format datasets so the model can learn efficiently and avoid catastrophic forgetting.

## Architecture Baseline

- Think before acting: identify the smallest correct implementation path, expected side effects, memory impact (VRAM), and rollback risk before changing code.
- Build modules as vertical feature slices under `src/`.
- Keep each module organized by responsibility. Place configuration validation in `src/config/`, data ingestion and preparation in `src/data/`, model loading and PEFT setup in `src/models/`, and training orchestration in `src/training/`.
- Do not leave mixed files directly in a module root.
- Keep entry points (in `scripts/`) thin; put orchestration in the `src` domain modules.
- Preserve the global project contract: strictly use `uv` for package management (`uv add`, `uv sync`, `uv run`). Never use traditional `pip` or `poetry`.
- **ALWAYS** consult the **Context7 MCP** whenever you use, download, or update a new library to ensure you are using the most current and correct API.
- Apply SOLID principles rigorously: functions and classes must have a single responsibility, depend on abstractions (Dependency Inversion), and be easily extensible for new model versions (Open/Closed).
- Keep the system intact: every change must preserve type safety, build configurations, and existing behavior.
- Do not leave modeled work blank: anything introduced in the configs must be used, configured, and implemented in the pipeline.
- Code must be completely self-explanatory. The use of code comments (`#` or `"""`) to explain logic is strictly forbidden. Write clean, readable code with descriptive names instead.

## AI Training Domain Baseline

- Treat the base model `unsloth/DeepSeek-R1-Distill-Qwen-14B` (Recruta 1.0 Lite) as the primary foundational model. Configurations reside in `configs/lite.yaml`.
- Treat model versioning as modular. New versions (e.g., Pro, Max) must inherit the base infrastructure and only create new `.yaml` files or dataset directories, without rewriting the primary training scripts.
- Treat Kaggle and Hugging Face as the core data sources.
- Use the following designated datasets:
  - **Kaggle:** `saugataroyarghya/resume-dataset`, `snehaanbhawal/resume-dataset`, `yashpwrr/resume-ner-training-dataset`, `mdtalhask/ai-powered-resume-screening-dataset-2025`, `thejohnwick001/resume-data-for-ranking`.
  - **Hugging Face:** `yashpwr/resume-ner-bert-v2`, `0xnbk/resume-ats-score-v1-en`, `netsol/resume-score-details`.
- Use the `datasets` library to load HF datasets (e.g., `load_dataset("0xnbk/resume-ats-score-v1-en")`).
- Design data pipelines so future parsing, formatting, and training loops can evolve without rewriting the module boundary.
