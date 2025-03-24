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
    Cria uma base de dados vetorial a partir dos documentos.

    Args:
        documents: Lista de documentos a serem indexados.
        embeddings: Modelo de embeddings a ser utilizado.
        persist_directory: Diretório para persistir o banco de vetores (opcional).

    Returns:
        Instância do Chroma DB.
    """
    # Se um diretório de persistência for fornecido, verifique se existe
    if persist_directory and not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    # Cria um banco Chroma com os documentos
    vectordb = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=persist_directory
    )

    # Persiste os vetores se um diretório for fornecido
    if persist_directory:
        vectordb.persist()
        print(f"Base de vetores persistida em {persist_directory}")

    return vectordb


def load_vector_store(persist_directory: str, embeddings: OpenAIEmbeddings) -> Chroma:
    """
    Carrega uma base de dados vetorial existente.

    Args:
        persist_directory: Diretório onde o banco de vetores está armazenado.
        embeddings: Modelo de embeddings a ser utilizado.

    Returns:
        Instância do Chroma DB.
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"O diretório {persist_directory} não existe.")

    print(f"Carregando base de vetores de {persist_directory}")

    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
