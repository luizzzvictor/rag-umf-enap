# RAG UMF/CNJ

Uma aplicação baseada em Retrieval Augmented Generation (RAG) para consulta a documentos da Unidade de Monitoramento e Fiscalização do Sistema Interamericano de Direitos Humanos (UMF/CNJ).

## Funcionalidades

- Acesso direto aos Relatórios Finais da UMF/CNJ pré-carregados
- Interface de chat para fazer perguntas sobre o conteúdo dos documentos
- Citação das fontes nas respostas
- Persistência de dados usando ChromaDB

## Requisitos

- Python 3.8+
- Dependências listadas em `requirements.txt`
- Chave de API da OpenAI

## Instalação

1. Clone o repositório:

   ```bash
   git clone [url-do-repositorio]
   cd [nome-do-diretorio]
   ```

2. Crie e ative um ambiente virtual:

   ```bash
   python -m venv venv

   # No Windows
   venv\Scripts\activate

   # No Linux/MacOS
   source venv/bin/activate
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure sua chave da API da OpenAI:
   - Renomeie o arquivo `.env.example` para `.env`
   - Edite o arquivo `.env` e insira sua chave de API

## Uso

1. Execute a aplicação:

   ```bash
   streamlit run app.py
   ```

2. Acesse a aplicação em seu navegador:

   - A aplicação será iniciada em `http://localhost:8501`

3. Interagindo com a aplicação:

   - A aplicação já vem com os Relatórios Finais da UMF/CNJ pré-carregados
   - Use o campo de chat para fazer perguntas sobre os documentos
   - O sistema fornecerá respostas com base no conteúdo dos relatórios

4. Persistência de dados:

   - Os Relatórios Finais são mantidos na pasta `data/pdfs`
   - Os vetores de embeddings são persistidos em `data/vectordb`
   - Ao reiniciar a aplicação, os dados são carregados automaticamente

5. Gerenciamento de dados:
   - Use o botão "Limpar Histórico" para apagar apenas o histórico de conversa

## Estrutura do Projeto

```
.
├── app.py                # Aplicação Streamlit principal
├── requirements.txt      # Dependências do projeto
├── utils/                # Módulos de utilidades
│   ├── __init__.py
│   ├── pdf_loader.py     # Carregamento e processamento de PDFs
│   ├── embeddings.py     # Funções para embeddings e armazenamento de vetores
│   ├── rag_chain.py      # Implementação da cadeia RAG
│   └── chat_memory.py    # Gerenciamento do histórico de conversas
└── data/                 # Diretório para armazenamento de dados
    ├── pdfs/             # Armazenamento dos Relatórios Finais pré-carregados
    └── vectordb/         # Armazenamento persistente da base vetorial
```

## Limitações

- O contexto de conversa é mantido apenas durante a sessão atual


## Solução de Problemas

### Erro de Conexão com o ChromaDB

Se você encontrar mensagens de erro relacionadas ao "tenant default_tenant" ou problemas de conexão com a base de dados:

1. A aplicação tentará resolver automaticamente o problema limpando e recriando a base de dados.
2. Se o problema persistir, use o botão "🔧 Reparar Base de Dados" que aparecerá na interface.
3. Após a reparação, a página será recarregada automaticamente.

