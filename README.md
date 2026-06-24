# Recruta AI Training

A scalable and corporate-grade Fine-Tuning pipeline for Large Language Models (LLMs) specifically targeted at the Human Resources (ATS - Applicant Tracking Systems) domain.

> [!NOTE]
> The current base model adopted for the "Lite" version is **`unsloth/DeepSeek-R1-Distill-Qwen-14B`**, focused on entity extraction (NER), scoring, and resume ranking.

## Architecture

This repository is built with strict adherence to SOLID principles, Clean Architecture, and type safety, ensuring modularity and maintainability.

```text
recruta-ai-training/
├── .agents/       # AI Training Skills and Guidelines (LoRA, PEFT, SOLID)
├── configs/       # Pydantic-validated YAML configurations
├── notebooks/     # Colab-ready Jupyter Notebooks for cloud GPU training
├── scripts/       # Thin CLI entrypoints
└── src/
    ├── config/    # Configuration parsing and validation
    ├── data/      # Multi-source dataset ingestion (Strategy Pattern)
    ├── models/    # Unsloth Model Loader and PEFT adapters
    ├── shared/    # Core Protocols and Typed Domain Errors
    └── training/  # SFTTrainer Orchestration and HuggingFace Publisher
```

## Features

- **Multi-Source Ingestion**: Seamless integration with Kaggle and Hugging Face APIs via the Strategy pattern.
- **Ultra-Fast Training**: Leveraging Unsloth for optimized VRAM usage and speed.
- **Automated Publishing**: Built-in publisher to merge 16-bit LoRA adapters and push directly to Hugging Face Hub.
- **Advanced PEFT**: Configured with `alpha >= 2 * r`, Cosine LR Scheduler, and customizable Warmup Ratios based on expert fine-tuning guidelines.
- **Type Safety**: Built with strict static typing, Pydantic validation, and structural pattern matching (Protocols).

## Prerequisites

- Python `>=3.10`
- [uv](https://github.com/astral-sh/uv) package manager
- Kaggle Account (for dataset downloading)
- Hugging Face Token with *Write* permission

## Getting Started

### Google Colab (Recommended)

The easiest way to run the end-to-end pipeline using hosted GPUs:

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `notebooks/recruta_colab_training.ipynb` file.
3. In the Colab Secrets tab, add your credentials:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
   - `HF_TOKEN`
4. Run all cells to automatically clone, ingest data, train the model, and publish it to Hugging Face.

### Local Development

> [!IMPORTANT]
> We strongly recommend using `uv` instead of `pip` to prevent Cuda/PyTorch dependency conflicts when using Unsloth.

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/willianOliveira-dev/recruta-ai-training.git
   cd recruta-ai-training
   uv sync
   ```

2. **Set Environment Variables:**
   Create a `.env` file or export them in your terminal:
   ```bash
   export KAGGLE_USERNAME="your_kaggle_username"
   export KAGGLE_KEY="your_kaggle_key"
   export HF_TOKEN="your_hf_write_token"
   ```

3. **Run the Pipeline:**
   ```bash
   # Optional: Only download and prepare data
   uv run scripts/prepare_data.py --config configs/lite.yaml

   # Train the model
   uv run scripts/train.py --config configs/lite.yaml

   # Merge 16-bit adapters and publish to Hugging Face
   uv run scripts/publish.py --config configs/lite.yaml
   ```

## Configuration

The entire system is configuration-driven via the `configs/` directory. You can add or modify data sources directly in the YAML file without touching the Python code.

```yaml
data:
  sources:
    - name: "custom-dataset"
      source_type: "kaggle"
      uri: "kaggle_user/dataset_name"
```
