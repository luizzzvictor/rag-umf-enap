"""
Módulo para geração e armazenamento de embeddings.
"""

import os
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def get_embeddings(api_key: Optional[str] = None) -> OpenAIEmbeddings:
    """
    Cria uma instância do modelo de embeddings.

    Args:
        api_key: Chave de API da OpenAI (opcional se já estiver no ambiente).

    Returns:
        Instância de OpenAIEmbeddings.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    return OpenAIEmbeddings(model="text-embedding-ada-002")


def create_vector_store(
    documents: List[Document],
    embeddings: OpenAIEmbeddings,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Cria uma nova base de vetores a partir dos documentos processados.

    Args:
        documents: Lista de documentos a serem indexados.
        embeddings: Modelo de embeddings a ser utilizado.
        persist_directory: Diretório para persistir o banco de vetores (opcional).

    Returns:
        Instância do Chroma DB.
    """
    try:
        # Configurações atualizadas conforme a migração do ChromaDB
        import chromadb

        # Criar cliente com a nova configuração
        if persist_directory:
            client = chromadb.PersistentClient(path=persist_directory)
        else:
            client = chromadb.Client()

        # Cria a base de vetores com o novo cliente
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            client=client,
        )

        # Persiste a base se um diretório for fornecido
        if persist_directory:
            if hasattr(vector_store, "persist"):
                vector_store.persist()
                print(f"Base de vetores persistida em {persist_directory}")

        return vector_store
    except Exception as e:
        print(f"Erro ao criar base de vetores: {str(e)}")
        # Se houve erro e temos um diretório de persistência, tenta recriá-lo
        if persist_directory and os.path.exists(persist_directory):
            try:
                import shutil

                shutil.rmtree(persist_directory)
                os.makedirs(persist_directory, exist_ok=True)
                print(
                    f"Diretório de persistência {persist_directory} recriado após erro"
                )
            except Exception as cleanup_error:
                print(f"Erro ao limpar diretório: {str(cleanup_error)}")

        # Relança o erro para ser tratado pelo chamador
        raise


def load_vector_store(persist_directory: str, embeddings: OpenAIEmbeddings) -> Chroma:
    """
    Carrega uma base de dados vetorial existente.

    Args:
        persist_directory: Diretório onde o banco de vetores está armazenado.
        embeddings: Modelo de embeddings a ser utilizado.

    Returns:
        Instância do Chroma DB.
    """
    try:
        # Configurações atualizadas conforme a migração do ChromaDB
        import chromadb

        # Criar cliente com a nova configuração
        client = chromadb.PersistentClient(path=persist_directory)

        print(f"Carregando base de vetores de {persist_directory}")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client=client,
        )
    except Exception as e:
        print(f"Erro ao carregar base de vetores: {str(e)}")
        # Relança o erro para ser tratado pelo chamador
        raise
