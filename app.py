"""
Aplicação RAG para consulta de documentos da UMF/CNJ
"""

import os
import shutil
import tempfile
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from utils.chat_memory import StreamlitChatHistory, get_conversation_memory
from utils.embeddings import create_vector_store, get_embeddings, load_vector_store
from utils.pdf_loader import process_pdf
from utils.rag_chain import create_rag_chain

# Carrega variáveis de ambiente
load_dotenv()

# Configuração da página
st.set_page_config(page_title="RAG UMF/CNJ", page_icon="📚", layout="wide")

# Configuração de diretórios
VECTOR_DB_DIR = os.path.join("data", "vectordb")
PDF_STORAGE_DIR = os.path.join("data", "pdfs")

# Cria os diretórios se não existirem
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# Inicialização da sessão
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


def initialize_llm():
    """Inicializa o modelo de linguagem."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        api_key = st.secrets.get("OPENAI_API_KEY", None)

    if not api_key:
        st.error(
            "Chave API da OpenAI não encontrada. Defina no arquivo .env ou nas secrets do Streamlit."
        )
        st.stop()

    return ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.2)


def initialize_vector_store():
    """Inicializa ou carrega o vector store persistente."""
    if st.session_state.vector_store is None:
        try:
            # Verifica se o diretório do vectordb já existe e tem conteúdo
            if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
                st.info("Carregando base de vetores existente...")
                embeddings = get_embeddings()
                st.session_state.vector_store = load_vector_store(
                    VECTOR_DB_DIR, embeddings
                )

                # Carrega a lista de arquivos processados
                if os.path.exists(PDF_STORAGE_DIR):
                    st.session_state.processed_files = [
                        f for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pdf")
                    ]

                st.success("Base de vetores carregada com sucesso!")
            else:
                st.info(
                    "Nenhuma base de vetores encontrada. Será criada uma nova quando documentos forem adicionados."
                )
        except Exception as e:
            st.error(f"Erro ao carregar base de vetores: {str(e)}")

    return st.session_state.vector_store


def initialize_rag_chain():
    """Inicializa a cadeia RAG se o vector store estiver disponível."""
    if st.session_state.vector_store and st.session_state.rag_chain is None:
        # Cria o histórico de chat
        chat_history = StreamlitChatHistory(st.session_state)
        memory = get_conversation_memory(chat_history)

        # Inicializa o modelo de linguagem
        llm = initialize_llm()

        # Cria a cadeia RAG
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        st.session_state.rag_chain = create_rag_chain(retriever, llm, memory)


def save_uploaded_pdf(uploaded_file):
    """Salva o PDF carregado na pasta de armazenamento."""
    file_path = os.path.join(PDF_STORAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_uploaded_file(uploaded_file):
    """Processa um arquivo PDF carregado."""
    # Verifica se o arquivo já foi processado
    if uploaded_file.name in st.session_state.processed_files:
        st.warning(f"O arquivo {uploaded_file.name} já foi processado.")
        return False

    try:
        with st.spinner(f"Processando {uploaded_file.name}..."):
            # Salva o PDF na pasta de armazenamento
            file_path = save_uploaded_pdf(uploaded_file)

            # Processa o PDF
            documents = process_pdf(file_path)

            if not documents:
                st.warning(f"Nenhum conteúdo extraído do arquivo {uploaded_file.name}.")
                return False

            # Inicializa embeddings
            embeddings = get_embeddings()

            # Cria ou atualiza o vector store
            if st.session_state.vector_store is None:
                st.session_state.vector_store = create_vector_store(
                    documents=documents,
                    embeddings=embeddings,
                    persist_directory=VECTOR_DB_DIR,
                )
            else:
                st.session_state.vector_store.add_documents(documents)
                st.session_state.vector_store.persist()  # Garante que os novos documentos sejam persistidos

            # Atualiza a lista de arquivos processados
            st.session_state.processed_files.append(uploaded_file.name)

            # Inicializa o RAG chain se ainda não existir
            initialize_rag_chain()

        st.success(f"Arquivo {uploaded_file.name} processado com sucesso!")
        return True

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")
        return False


def reset_chat():
    """Reseta o histórico de chat."""
    if "chat_messages" in st.session_state:
        st.session_state.chat_messages = []


def clear_all_data():
    """Limpa todos os dados do vectordb e PDFs armazenados."""
    try:
        # Limpa a vectordb
        if os.path.exists(VECTOR_DB_DIR):
            shutil.rmtree(VECTOR_DB_DIR)
            os.makedirs(VECTOR_DB_DIR)

        # Remove os PDFs armazenados
        if os.path.exists(PDF_STORAGE_DIR):
            for file in os.listdir(PDF_STORAGE_DIR):
                if file.endswith(".pdf"):
                    os.remove(os.path.join(PDF_STORAGE_DIR, file))

        # Reseta as variáveis de sessão
        st.session_state.processed_files = []
        st.session_state.vector_store = None
        st.session_state.rag_chain = None

        # Limpa o histórico de chat
        reset_chat()

        return True
    except Exception as e:
        st.error(f"Erro ao limpar dados: {str(e)}")
        return False


def main():
    # Título
    st.title("📚 RAG UMF/CNJ")
    st.subheader(
        "Consulta de documentos da Unidade de Monitoramento e Fiscalização do Sistema Interamericano de Direitos Humanos"
    )

    # Inicializa ou carrega a base de vetores persistente
    initialize_vector_store()

    # Inicializa a cadeia RAG se necessário
    if st.session_state.vector_store:
        initialize_rag_chain()

    # Barra lateral
    with st.sidebar:
        st.header("📄 Carregar Documentos")

        # Upload de arquivos
        uploaded_files = st.file_uploader(
            "Selecione os PDFs da UMF/CNJ", type="pdf", accept_multiple_files=True
        )

        # Botão para processar os arquivos
        if uploaded_files:
            if st.button("Processar Documentos"):
                for uploaded_file in uploaded_files:
                    process_uploaded_file(uploaded_file)

        # Lista de arquivos processados
        if st.session_state.processed_files:
            st.write("---")
            st.subheader("Documentos Processados:")
            for filename in st.session_state.processed_files:
                st.write(f"- {filename}")

        # Botão para limpar histórico
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Limpar Histórico"):
                reset_chat()
                st.success("Histórico de conversa limpo!")

        with col2:
            if st.button("Limpar Tudo", type="primary", use_container_width=True):
                if clear_all_data():
                    st.success("Todos os dados foram limpos!")

    # Área principal
    if not st.session_state.processed_files:
        st.info("👈 Carregue os documentos da UMF/CNJ no painel lateral para começar.")
        st.stop()

    # Interface de chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Exibe mensagens
    for message in st.session_state.chat_messages:
        if message.type == "human":
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

    # Campo de entrada de texto
    if prompt := st.chat_input("Digite sua pergunta sobre os documentos..."):
        # Adiciona a mensagem do usuário
        with st.chat_message("user"):
            st.write(prompt)

        # Adiciona ao histórico
        chat_history = StreamlitChatHistory(st.session_state)
        chat_history.add_user_message(prompt)

        # Gera a resposta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = st.session_state.rag_chain.invoke({"question": prompt})
                st.write(response)

        # Adiciona ao histórico
        chat_history.add_ai_message(response)


if __name__ == "__main__":
    main()
