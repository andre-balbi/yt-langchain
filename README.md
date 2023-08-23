# Aplicação utilizando a biblioteca LangChain em Vídeos do YouTube

#### Introdução

O LangChain pode ser empregado para construir sistemas inteligentes semelhantes ao Auto-GPT. Estou seguro de que neste momento se apresenta uma oportunidade incrível para cientistas de dados e engenheiros de IA se destacarem, explorando essas ferramentas de maneira proveitosa. Podendo aproveitar essas ferramentas inovadoras, entrando um campo promissor e promovendo o desenvolvimento de soluções avançadas. A aplicação do LangChain abre portas para a criação de sistemas que não apenas atendam às necessidades atuais, mas também quebrando fronteiras na inteligência artificial, impulsionando a próxima geração de inovações tecnológicas.

Neste projeto, exploramos a utilização da biblioteca LangChain no desenvolvimento de aplicações. Utilizamos modelos de linguagem, como o GPT-3.5 Turbo da OpenAI, para criar um banco de dados de pesquisa a partir de transcrições de vídeos do YouTube. Além disso, analisamos as buscas de similaridade usando a biblioteca FAISS, com base nas perguntas dos usuários. Nosso foco é oferecer respostas precisas e relevantes a essas perguntas, melhorando a interação do usuário com as informações nos vídeos

![overview](img/overview.jpg)

#### Objetivos

- Utilização da biblioteca LangChain para desenvolver aplicações.
- Criação de banco de dados pesquisável de transcrições de vídeos do YouTube.
- Análise de similaridade com a biblioteca FAISS.
- Fornecimento de respostas precisas e pertinentes a perguntas.

#### Instalação

1. Clonar the repository

```bash
git clone <https://github.com/daveebbelaar/langchain-experiments.git>

```

2. Criar um ambiente Python

`Python 3.6` ou superior usando `venv` ou `conda`. Usando `conda`:

```bash
conda create -n langchain-env python=3.8
```

3. Instalar as dependências necessárias

```bash
pip install -r requirements.txt

```

4. Configurar as chaves em um arquivo .env**

Crie um arquivo **`.env`** na pasta raiz do projeto. No interior do arquivo, adicione a chave API da OpenAI:

```makefile
OPENAI_API_KEY="your_api_key_here"

```

Carregue o arquivo .env usando o seguinte código:

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```
---

### Primeiro Contato com Langchain

No link abaixo, encontra-se um notebook com exemplos práticos que serve como uma forma de introdução ao conceito da biblioteca LangChain. Esses exemplos visam facilitar a compreensão da aplicação real que será desenvolvida.

[`Quickstart Guide`](introduction/quickstart_guide.ipynb)

---

### Códigos do Projeto

Códigos originais: [`youtube_chat.py`](youtube/youtube_chat.py)

Passo a passo explicando a contrução da aplicação: [`youtube_chat.ipynb`](youtube/youtube_chat.ipynb)
