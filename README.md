# Recruta AI - LLM Fine-Tuning Repository

Repositório corporativo e estruturado com base nos princípios **SOLID** para treinamento, fine-tuning e avaliação de modelos baseados em LLM focados em análise de currículos (Resumes).

## 📂 Arquitetura do Projeto

O repositório foi projetado visando escalabilidade para novas versões (Pro, Max) a partir da nossa base Lite, permitindo configurações sobrepostas e fácil adição de novos datasets.

```text
recruta-ai-training/
├── @AGENTS.md                  # Diretrizes para Agentes IA e restrições arquiteturais
├── README.md                   # Documentação base do projeto e tree view
├── pyproject.toml              # Dependências gerenciadas via `uv`
│
├── configs/                    # Camada de configuração (YAML)
│   ├── base.yaml               # Parâmetros gerais globais
│   ├── lite.yaml               # Recruta 1.0 Lite (unsloth/DeepSeek-R1-Distill-Qwen-14B)
│   └── (futuros).yaml          # Pro, Max, etc.
│
├── data/                       # Armazenamento de dados locais (se necessário)
│   ├── raw/                    # Datasets originais do Kaggle/HF (Ignorados no git)
│   └── processed/              # Datasets processados no formato de instrução SFT
│
├── src/                        # Código base focado em SOLID
│   ├── __init__.py
│   ├── config/                 # Pydantic Settings para validação das configs
│   ├── data/                   # Ingestão e preparação de dados (Kaggle e Hugging Face)
│   ├── models/                 # Gerenciamento e carregamento de modelos (Unsloth)
│   └── training/               # Lógica de fine-tuning (SFTTrainer e afins)
│
└── scripts/                    # CLI Scripts / Entrypoints
    ├── prepare_data.py         # Ingestão e formatação dos datasets
    └── train.py                # Script principal de treinamento (recebe o config como arg)
```

## 🛠️ Tecnologias Principais
- **[Unsloth](https://github.com/unslothai/unsloth):** Para fine-tuning otimizado, veloz e com menor uso de VRAM de modelos grandes.
- **[Hugging Face `datasets` e `transformers`](https://huggingface.co/):** Pipeline de dados e inferência.
- **[Pydantic](https://docs.pydantic.dev/):** Validação de configurações (Config driven development).
- **[uv](https://github.com/astral-sh/uv):** Gerenciador de pacotes ultra-rápido para Python.

## 🚀 Como Iniciar

1. **Instale o `uv`** (caso não tenha):
   ```bash
   pip install uv
   ```

2. **Crie o ambiente virtual e instale as dependências**:
   ```bash
   uv venv
   # No Windows: .\.venv\Scripts\activate
   # No Linux/Mac: source .venv/bin/activate
   uv pip install -e .
   ```

3. **Prepara os Dados**:
   ```bash
   python scripts/prepare_data.py --config configs/lite.yaml
   ```

4. **Inicie o Treinamento**:
   ```bash
   python scripts/train.py --config configs/lite.yaml
   ```
