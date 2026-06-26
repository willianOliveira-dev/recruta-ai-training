# Recruta AI Training

Uma pipeline corporativa e escalável para *Fine-Tuning* de Modelos de Linguagem Grande (LLMs) voltada especificamente para o domínio de Recursos Humanos (ATS - *Applicant Tracking Systems*).

> [!NOTE]
> O modelo base atual adotado para a versão "Micro" é o **`unsloth/Llama-3.2-3B-Instruct`**, focado em extração de entidades (NER), *scoring* e ranqueamento de currículos. Modelo leve o suficiente para rodar gratuitamente no Hugging Face Spaces (CPU Basic, 16GB RAM).

## Arquitetura

Este repositório foi construído com adesão estrita aos princípios SOLID, Clean Architecture e segurança de tipos (Type Safety), garantindo modularidade e manutenibilidade.

```text
recruta-ai-training/
├── .agents/       # Skills e Diretrizes de Treinamento IA (LoRA, PEFT, SOLID)
├── configs/       # Configurações YAML validadas via Pydantic
├── notebooks/     # Jupyter Notebooks prontos para treinamento com GPU em nuvem (Colab)
├── scripts/       # Pontos de entrada CLI simplificados (Thin entrypoints)
└── src/
    ├── config/    # Leitura e validação das configurações
    ├── data/      # Ingestão de datasets de múltiplas fontes (Padrão Strategy)
    ├── models/    # Carregador de modelos Unsloth e adaptadores PEFT
    ├── shared/    # Protocolos centrais e Erros de Domínio Tipados
    └── training/  # Orquestração do SFTTrainer e Publicador no HuggingFace
```

## Funcionalidades

- **Ingestão Multi-Fonte**: Integração fluida com APIs do Hugging Face via padrão Strategy.
- **Treinamento Ultra-Rápido**: Utiliza o Unsloth para otimização de velocidade e uso de memória VRAM.
- **Exportação GGUF**: Compila o modelo treinado no formato `.gguf` (quantizado `q4_k_m`), pronto para rodar com Ollama em qualquer infraestrutura com CPU.
- **PEFT Avançado**: Configurado com `alpha >= 2 * r`, *Cosine LR Scheduler*, e *Warmup Ratios* customizáveis, baseados nas melhores práticas e guias de especialistas em fine-tuning.
- **Segurança de Tipos (Type Safety)**: Construído com tipagem estática rigorosa, validação Pydantic e correspondência de padrões estruturais (Protocols).

## Pré-requisitos

- Python `>=3.10`
- Gerenciador de pacotes [uv](https://github.com/astral-sh/uv)


## Como Iniciar

### Google Colab (Recomendado)

A maneira mais fácil de executar o pipeline de ponta a ponta usando GPUs em nuvem:

1. Abra o [Google Colab](https://colab.research.google.com/).
2. Faça o upload do arquivo `notebooks/recruta_colab_training.ipynb`.
3. Na aba Secrets do Colab, adicione suas credenciais:
   - `HF_TOKEN`
4. Execute todas as células para clonar o repositório, ingerir os dados e treinar o modelo de forma automatizada.

### Desenvolvimento Local

> [!IMPORTANT]
> Recomendamos fortemente usar o `uv` no lugar do `pip` para prevenir conflitos nas dependências de Cuda/PyTorch ao utilizar o Unsloth.

1. **Clone o repositório e instale as dependências:**
   ```bash
   git clone https://github.com/willianOliveira-dev/recruta-ai-training.git
   cd recruta-ai-training
   uv sync
   ```

2. **Execute o Pipeline:**
   ```bash
   # Opcional: Apenas fazer o download e preparar os dados
   uv run scripts/prepare_data.py --config configs/micro.yaml

   # Treinar o modelo
   uv run scripts/train.py --config configs/micro.yaml
   ```

## Configuração

Todo o sistema é orientado por configuração através do diretório `configs/`. Você pode adicionar ou modificar as fontes de dados diretamente no arquivo YAML, sem tocar no código em Python.

```yaml
data:
  sources:
```
