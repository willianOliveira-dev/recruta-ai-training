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

- Treat the base model `unsloth/Llama-3.2-3B-Instruct` (Recruta 1.0 Micro) as the primary foundational model. Configurations reside in `configs/micro.yaml`.
- Treat model versioning as modular. New versions (e.g., Lite, Pro, Max) must inherit the base infrastructure and only create new `.yaml` files or dataset directories, without rewriting the primary training scripts.
- Treat Hugging Face as the primary data source. All datasets must be loadable via `load_dataset()` to work in-memory during training.
- Use the following designated datasets:
  - `yashpwr/resume-ner-training-data` — NER labels for resume entities (skills, education, experience).
  - `0xnbk/resume-ats-score-v1-en` — ATS compatibility scores between resumes and job descriptions.
  - `sandeeppanem/resume-json-extraction-5k` — structured JSON extraction from raw resume text (instruction/output format).
  - `datasetmaster/resumes` — corpus of real and synthetic anonymized resumes.
  - `LlamaFactoryAI/cv-job-description-matching` — CV and job description matching pairs for instruction tuning.
- Use the `datasets` library to load HF datasets (e.g., `load_dataset("0xnbk/resume-ats-score-v1-en")`).
- Design data pipelines so future parsing, formatting, and training loops can evolve without rewriting the module boundary.
