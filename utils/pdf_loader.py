"""
Módulo para carregamento e processamento de arquivos PDF.
"""

import os
import re
from typing import Any, Dict, List

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extrair_texto_pdf(file_path: str) -> str:
    """
    Extrai texto de um PDF usando PyMuPDF com tratamento aprimorado.

    Args:
        file_path: Caminho para o arquivo PDF.

    Returns:
        String com o texto extraído e pré-processado.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")

    texto_final = ""
    try:
        with fitz.open(file_path) as doc:
            for pagina in doc:
                texto = pagina.get_text("text")
                texto_final += texto.replace("\n", " ") + "\n\n"
        return texto_final.strip()
    except Exception as e:
        print(f"Erro ao extrair texto do PDF {file_path}: {str(e)}")
        return ""


def limpar_texto(texto: str) -> str:
    """
    Limpa e formata o texto extraído do PDF.

    Args:
        texto: Texto bruto extraído do PDF.

    Returns:
        Texto limpo e formatado.
    """
    # Remove múltiplos espaços e quebras de linha desnecessárias
    texto_limpo = re.sub(r"\s+", " ", texto).strip()

    # Reinserir quebra dupla onde houver pontuação seguida por maiúscula
    # (padrão típico de novo parágrafo)
    texto_limpo = re.sub(r"([.!?]) ([A-ZÀ-Ú])", r"\1\n\n\2", texto_limpo)

    return texto_limpo


def load_pdf(file_path: str) -> List[Document]:
    """
    Carrega um arquivo PDF e extrai seu conteúdo com pré-processamento aprimorado.

    Args:
        file_path: Caminho para o arquivo PDF.

    Returns:
        Lista de documentos extraídos do PDF.
    """
    try:
        # Extrai e limpa o texto
        texto_bruto = extrair_texto_pdf(file_path)
        if not texto_bruto:
            return []

        texto_limpo = limpar_texto(texto_bruto)

        # Cria um único documento com o texto limpo
        documento = Document(
            page_content=texto_limpo, metadata={"source": os.path.basename(file_path)}
        )

        return [documento]
    except Exception as e:
        print(f"Erro ao carregar o PDF {file_path}: {str(e)}")
        return []


def split_documents(
    documents: List[Document], chunk_size: int = 2000, chunk_overlap: int = 200
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
        separators=["\n\n", ".", "!", "?", " ", ""],
    )

    return text_splitter.split_documents(documents)


def process_pdf(
    file_path: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[Document]:
    """
    Processa um arquivo PDF: carrega, limpa e divide em chunks.

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
