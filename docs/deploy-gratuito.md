# Deploy Gratuito da API do Recruta AI

Após o treinamento do modelo "Micro" (Llama-3.2-3B) e sua respectiva compilação para o formato leve (`.gguf`), você pode hospedar o modelo **100% de graça e sem limites de requisição** diretamente no Hugging Face Spaces.

## Passo a Passo: Hospedando no Hugging Face Spaces (CPU Free)

A infraestrutura do Hugging Face oferece máquinas gratuitas (CPU Basic com 16GB de RAM). Como compilamos nosso modelo no padrão `q4_k_m`, ele consome cerca de 3 a 4GB, rodando com folga neste plano.

1. **Crie um novo Space:**
   - Acesse [Hugging Face Spaces](https://huggingface.co/spaces) e clique em **"Create new Space"**.
   - **Space name:** `recruta-ai-api`.
   - **License:** Escolha uma de sua preferência (ex: MIT).
   - **Select the Space SDK:** Escolha **Docker** -> **Blank**.
   - **Space Hardware:** Escolha **CPU Basic - 2 vCPU · 16 GB - Free**.

2. **Crie o arquivo Dockerfile:**
   Assim que o Space for criado, vá na aba "Files" e crie um arquivo chamado exatamente `Dockerfile` com o seguinte conteúdo:

   ```dockerfile
   # Usamos a imagem oficial do Ollama que já expõe a API padrão (compatível com OpenAI)
   FROM ollama/ollama:latest

   # Baixamos o modelo GGUF diretamente do nosso repositório no Hugging Face (Altere para o SEU repositório de pesos)
   # O comando abaixo assume que o modelo treinado já está no seu Hugging Face (o publicador faz isso por você).
   RUN apt-get update && apt-get install -y curl
   RUN curl -L -o /root/recruta.gguf "https://huggingface.co/SEU-USUARIO/recruta-ats-micro-3b-gguf/resolve/main/unsloth.Q4_K_M.gguf"

   # Criamos um script simples para injetar o modelo no Ollama durante a inicialização
   RUN echo 'FROM /root/recruta.gguf' > /root/Modelfile
   
   # O Hugging Face Spaces mapeia a porta 7860
   ENV OLLAMA_HOST=0.0.0.0:7860
   
   EXPOSE 7860

   CMD ollama serve & sleep 5 && ollama create recruta -f /root/Modelfile && wait
   ```

3. **Inicie o Servidor:**
   Ao salvar (Commit) o `Dockerfile`, o Hugging Face começará o "Building" automaticamente. Isso leva de 2 a 5 minutos. Após o status mudar para **"Running"**, sua API está no ar!

## Consumindo na sua API NestJS

O seu Space vai gerar uma URL de Host direta, no formato `https://seu-usuario-recruta-ai-api.hf.space`.

No seu NestJS, basta você utilizar o pacote oficial da OpenAI (já que o Ollama emula a API) ou fazer requisições HTTP REST (Fetch/Axios) diretamente para essa URL:

```typescript
import OpenAI from 'openai';

const aiClient = new OpenAI({
  baseURL: 'https://seu-usuario-recruta-ai-api.hf.space/v1',
  apiKey: 'nenhuma', // Ollama não exige autenticação por padrão
});

async function analisarCurriculo(texto: string) {
  const response = await aiClient.chat.completions.create({
    model: 'recruta', // Nome que criamos no Dockerfile
    messages: [{ role: 'user', content: texto }],
  });
  return response.choices[0].message.content;
}
```

> [!TIP]
> A resposta pode demorar alguns segundos por ser processada por uma CPU gratuita (sem placa de vídeo dedicada), mas será gerada com **precisão absoluta**, embasada no nosso _Fine-Tuning_ corporativo voltado especificamente para leitura e ranqueamento de currículos, a custo zero de hospedagem.
