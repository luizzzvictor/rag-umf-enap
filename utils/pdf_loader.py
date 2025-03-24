"""
Módulo para carregamento e processamento de arquivos PDF.
"""

import os
from typing import Any, Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> List[Document]:
    """
    Carrega um arquivo PDF e extrai seu conteúdo.

    Args:
        file_path: Caminho para o arquivo PDF.

    Returns:
        Lista de documentos extraídos do PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Adiciona metadados sobre a fonte
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)

        return documents
    except Exception as e:
        print(f"Erro ao carregar o PDF {file_path}: {str(e)}")
        return []


def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Divide os documentos em chunks menores para processamento.

    Args:
        documents: Lista de documentos a serem divididos.
        chunk_size: Tamanho de cada chunk em caracteres.
        chunk_overlap: Quantidade de sobreposição entre chunks em caracteres.

    Returns:
        Lista de documentos divididos em chunks.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    return text_splitter.split_documents(documents)


def process_pdf(
    file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Processa um arquivo PDF: carrega e divide em chunks.

    Args:
        file_path: Caminho para o arquivo PDF.
        chunk_size: Tamanho de cada chunk em caracteres.
        chunk_overlap: Quantidade de sobreposição entre chunks em caracteres.

    Returns:
        Lista de documentos processados.
    """
    try:
        documents = load_pdf(file_path)
        if not documents:
            print(f"Nenhum conteúdo extraído do PDF {file_path}")
            return []

        chunks = split_documents(documents, chunk_size, chunk_overlap)
        return chunks
    except Exception as e:
        print(f"Erro ao processar o PDF {file_path}: {str(e)}")
        return []
