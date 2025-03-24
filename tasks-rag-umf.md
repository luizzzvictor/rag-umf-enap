# Plano para desenvolvimento de RAG com Chatbot da UMF/CNJ

## Visão Geral

Este plano detalha a criação de um Minimum Viable Product (MVP) para um sistema de Retrieval Augmented Generation (RAG) com funcionalidade de chatbot que consultará PDFs de publicações da Unidade de Monitoramento e Fiscalização do Sistema Interamericano de Direitos Humanos (UMF/CNJ). O sistema manterá o contexto da conversa entre o usuário e o modelo.

## Objetivo

Desenvolver uma aplicação web simples que permita aos usuários:

1. Fazer upload de PDFs da UMF/CNJ
2. Fazer perguntas sobre o conteúdo desses documentos
3. Receber respostas contextualizadas baseadas nos dados dos PDFs
4. Manter um histórico de conversas e contexto para perguntas de acompanhamento

## Estrutura do Projeto

```
rag-umf-cnj/
├── app.py              # Aplicação Streamlit principal
├── requirements.txt    # Dependências do projeto
├── utils/
│   ├── __init__.py
│   ├── pdf_loader.py   # Funções para carregamento de PDFs
│   ├── embeddings.py   # Funções relacionadas aos embeddings
│   ├── rag_chain.py    # Implementação da cadeia RAG
│   └── chat_memory.py  # Gerenciamento do histórico de conversas
└── data/               # Pasta para armazenar os PDFs e vetores
    ├── pdfs/           # Armazenamento dos PDFs carregados
    └── vectordb/       # Armazenamento persistente da base vetorial
```

## Tarefas de Implementação

### Etapa 1: Configuração do Ambiente

- [x] 1.1. Criar e ativar ambiente virtual
- [x] 1.2. Criar estrutura de diretórios do projeto
- [x] 1.3. Instalar bibliotecas necessárias e criar requirements.txt
  - streamlit
  - langchain
  - langchain-openai (ou outro provedor LLM)
  - langchain-community
  - langchain-chroma
  - pypdf
  - chromadb (para armazenamento de vetores)
  - tiktoken
  - python-dotenv (para gerenciar variáveis de ambiente)

### Etapa 2: Implementação da Funcionalidade de Processamento de PDFs

- [x] 2.1. Criar função para carregar PDFs usando LangChain
- [x] 2.2. Implementar divisão do texto em chunks apropriados
- [x] 2.3. Criar função para gerar e armazenar embeddings dos chunks
- [x] 2.4. Implementar sistema de gerenciamento de arquivos carregados

### Etapa 3: Implementação da Cadeia RAG com Memória de Conversação

- [x] 3.1. Configurar conexão com o modelo de linguagem (OpenAI, Azure, etc.)
- [x] 3.2. Implementar prompt template para consulta RAG
- [x] 3.3. Criar a cadeia de RAG completa (recuperação + geração)
- [x] 3.4. Implementar sistema de memória para manter o contexto da conversa
- [x] 3.5. Integrar a memória da conversa com a cadeia RAG

### Etapa 4: Desenvolvimento da Interface Streamlit

- [x] 4.1. Criar a estrutura básica da aplicação
- [x] 4.2. Implementar upload de PDFs
- [x] 4.3. Implementar campo de consulta e exibição de respostas
- [x] 4.4. Adicionar visualização do histórico de conversas
- [x] 4.5. Implementar elementos de UI para melhorar a experiência do usuário
- [x] 4.6. Adicionar funcionalidade para limpar histórico/iniciar nova conversa

### Etapa 5: Implementação da Persistência de Dados

- [x] 5.1. Configurar persistência de vectorstore com ChromaDB
- [x] 5.2. Implementar armazenamento persistente de PDFs
- [x] 5.3. Criar sistema de carregamento automático da base existente
- [x] 5.4. Adicionar funcionalidade para limpar todos os dados

### Etapa 6: Testes e Otimização

- [ ] 6.1. Testar com PDFs reais da UMF/CNJ
- [ ] 6.2. Otimizar parâmetros (tamanho dos chunks, overlap, etc.)
- [ ] 6.3. Melhorar a interface do usuário com base em feedback
- [ ] 6.4. Testar a capacidade de manter contexto em conversas longas

## Detalhamento do MVP

### Funcionalidades do MVP

- Upload de PDFs
- Processamento dos PDFs para extração de texto e geração de embeddings
- Interface de chatbot para consultas
- Geração de respostas com citação das fontes
- Manutenção do contexto da conversa para perguntas de acompanhamento
- Persistência de dados entre sessões usando ChromaDB

### Limitações do MVP

- Contexto de conversa limitado ao tamanho da memória do modelo
- Sem autenticação de usuários
- Quantidade limitada de PDFs que podem ser processados simultaneamente

## Expansões Futuras

- Metadados mais ricos para os documentos
- Autenticação de usuários
- Histórico de consultas salvo entre sessões
- Visualização de citações diretas no PDF
- Suporte para outros formatos além de PDF
- Melhorias na gestão do contexto de conversa para conversas muito longas

## Como Usar o Plano

Este plano serve como um guia para implementação passo a passo com assistência de um agente de IA. Para cada tarefa:

1. Marque com [X] quando completada
2. Desenvolva a funcionalidade específica
3. Teste antes de avançar para a próxima tarefa
4. Revise e refine conforme necessário
