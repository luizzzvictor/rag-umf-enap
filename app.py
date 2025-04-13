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
st.set_page_config(
    page_title="RAG UMF/CNJ",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 350px;
    }   
    </style>
    """,
    unsafe_allow_html=True,
)

# Configuração de diretórios
VECTOR_DB_DIR = os.path.join("data", "vectordb")
PDF_STORAGE_DIR = os.path.join("data", "pdfs")

# Cria os diretórios se não existirem
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# Inicialização da sessão
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "document_metadata" not in st.session_state:
    st.session_state.document_metadata = {}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "data_cleared" not in st.session_state:
    st.session_state.data_cleared = False

# Adicionamos flag para rastrear erros de tenant
if "tenant_error_detected" not in st.session_state:
    st.session_state.tenant_error_detected = False


def force_clean_vectordb():
    """Força a limpeza do diretório vectordb para resolver problemas de acesso e tenant."""
    if not os.path.exists(VECTOR_DB_DIR):
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        return True

    try:
        # Tenta limpar o diretório recursivamente
        print(f"Forçando limpeza da base de dados em {VECTOR_DB_DIR}")
        st.toast("Recriando base de dados para resolver problemas...", icon="🔄")

        try:
            shutil.rmtree(VECTOR_DB_DIR)
        except Exception as e:
            print(f"Erro ao remover diretório: {str(e)}")

            # No Windows, forçar coleta de lixo pode ajudar a liberar handles de arquivos
            import gc

            gc.collect()

            # Se falhar, tenta deletar arquivos individualmente
            for file_name in os.listdir(VECTOR_DB_DIR):
                file_path = os.path.join(VECTOR_DB_DIR, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    print(f"Removido: {file_path}")
                except Exception as e:
                    print(f"Erro ao remover {file_path}: {str(e)}")

        # Recria o diretório limpo
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        print("Diretório do vectordb limpo e recriado com sucesso.")
        st.toast("Base de dados recriada com sucesso!", icon="✅")
        return True
    except Exception as e:
        print(f"Falha ao limpar vectordb: {str(e)}")
        st.toast(f"Erro ao recriar base de dados: {str(e)}", icon="❌")
        return False


def cleanup_orphaned_vectordb():
    """
    Verifica e limpa diretórios de vetores órfãos em caso de reinício inesperado.
    Esta função é executada no início da aplicação para garantir que
    arquivos de sessões anteriores não causem conflitos.
    """
    try:
        # Verificar se há uma flag de erro de tenant
        tenant_error_detected = False
        error_file_path = os.path.join(VECTOR_DB_DIR, "tenant_error.flag")

        # Verifica se o arquivo de flag existe (indicando erro anterior)
        if os.path.exists(error_file_path):
            tenant_error_detected = True

        # Se há arquivos no diretório, mas nenhuma conexão ativa
        if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
            # Se detectou erro de tenant ou o diretório parece corrompido
            if tenant_error_detected:
                # Limpa completamente
                force_clean_vectordb()
                return True

            # Mesmo sem flag, verifica se os arquivos estão acessíveis
            all_ok = True
            for root, dirs, files in os.walk(VECTOR_DB_DIR):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        # Tenta abrir o arquivo em modo de escrita
                        with open(file_path, "a"):
                            pass
                    except (IOError, PermissionError):
                        all_ok = False
                        break

            # Se algum arquivo não está acessível, tenta limpeza
            if not all_ok:
                force_clean_vectordb()
                return True
    except Exception as e:
        print(f"Erro ao verificar vectordb órfão: {e}")

    return False


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

    return ChatOpenAI(api_key=api_key, model="gpt-4o")


def initialize_vector_store():
    """Inicializa ou carrega o vector store persistente."""
    if st.session_state.vector_store is None:
        try:
            # Verifica se o diretório do vectordb já existe e tem conteúdo
            if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
                # Usa toast em vez de info para notificação de carregamento
                loading_toast = st.toast(
                    "Carregando base de vetores existente...", icon="🔄"
                )
                embeddings = get_embeddings()
                try:
                    # Usa o novo cliente do ChromaDB
                    import chromadb

                    st.session_state.vector_store = load_vector_store(
                        VECTOR_DB_DIR, embeddings
                    )

                    # Carrega a lista de arquivos processados
                    if os.path.exists(PDF_STORAGE_DIR):
                        st.session_state.processed_files = [
                            f for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pdf")
                        ]

                        # Verifica se é necessário gerar metadados para documentos existentes
                        # Isso acontece se estamos carregando documentos processados antes da implementação dos metadados
                        llm = None
                        for file_name in st.session_state.processed_files:
                            if file_name not in st.session_state.document_metadata:
                                file_path = os.path.join(PDF_STORAGE_DIR, file_name)

                                # Processa o documento para extrair texto
                                documents = process_pdf(file_path)

                                if documents:
                                    # Inicializa o LLM apenas uma vez se necessário
                                    if llm is None:
                                        llm = initialize_llm()

                                    # Extrair todo o texto do documento
                                    document_text = " ".join(
                                        [doc.page_content for doc in documents]
                                    )

                                    # Gerar título e resumo
                                    title, summary = extract_document_metadata(
                                        document_text, file_name, llm
                                    )

                                    # Armazenar os metadados
                                    st.session_state.document_metadata[file_name] = {
                                        "title": title,
                                        "summary": summary,
                                        "file_path": file_path,
                                    }

                    # Substitui o toast de carregamento por um de sucesso
                    loading_toast.toast(
                        "Base de vetores carregada com sucesso!", icon="✅"
                    )
                except Exception as e:
                    error_str = str(e)
                    # Detecta o erro específico de tenant
                    if (
                        "tenant default_tenant" in error_str
                        and "Could not connect" in error_str
                    ):
                        st.session_state.tenant_error_detected = True
                        # Cria um arquivo de flag para indicar o erro de tenant
                        error_flag_path = os.path.join(
                            VECTOR_DB_DIR, "tenant_error.flag"
                        )
                        with open(error_flag_path, "w") as f:
                            f.write(f"Error detected: {error_str}")

                        # Força limpeza e reinicia a aplicação
                        force_clean_vectordb()
                        st.toast(
                            "Detectado problema na base de vetores. Limpeza forçada realizada.",
                            icon="🔧",
                        )
                        st.warning(
                            "Base de dados vetorial corrompida. Foi necessário limpar os dados. Por favor, recarregue seus documentos."
                        )
                        st.session_state.vector_store = None
                    else:
                        # Para outros erros
                        st.toast(
                            f"Erro ao carregar base de vetores: {error_str}", icon="❌"
                        )
                        st.error(f"Erro ao carregar base de vetores: {error_str}")
            else:
                st.toast(
                    "Nenhuma base de vetores encontrada. Será criada quando documentos forem adicionados.",
                    icon="ℹ️",
                )
        except Exception as e:
            st.toast(f"Erro ao carregar base de vetores: {str(e)}", icon="❌")
            st.error(f"Erro ao carregar base de vetores: {str(e)}")

    return st.session_state.vector_store


def initialize_rag_chain(vector_store):
    """Inicializa a cadeia RAG se o vector store estiver disponível."""
    if vector_store and st.session_state.rag_chain is None:
        # Garante que o histórico de chat esteja inicializado
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Cria o histórico de chat
        chat_history = StreamlitChatHistory(st.session_state)
        memory = get_conversation_memory(chat_history)

        # Inicializa o modelo de linguagem
        llm = initialize_llm()

        # Cria a cadeia RAG
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 15}
        )
        st.session_state.rag_chain = create_rag_chain(retriever, llm, memory)


def save_uploaded_pdf(uploaded_file):
    """Salva o PDF carregado na pasta de armazenamento."""
    file_path = os.path.join(PDF_STORAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def extract_document_metadata(document_content, file_name, llm=None):
    """
    Extrai título significativo e resumo do documento.

    Args:
        document_content: Conteúdo extraído do documento
        file_name: Nome do arquivo original
        llm: Modelo de linguagem opcional

    Returns:
        Tupla (título, resumo) ambos como strings
    """
    # Se não temos conteúdo suficiente, retornamos valores padrão
    if not document_content or len(document_content) < 50:
        return (
            str(file_name).replace(".pdf", ""),
            "Não foi possível gerar um resumo deste documento.",
        )

    # Se o LLM não foi fornecido, inicialize-o
    if llm is None:
        llm = initialize_llm()

    try:
        # Pegar o início do documento para análise (primeiros 2000 caracteres)
        texto_inicial = (
            document_content[:8000]
            if len(document_content) > 8000
            else document_content
        )

        # Prompt para extrair título e resumo
        prompt = f"""
        Analise o início deste Relatório Final e extraia:
        1. O título de Relatório UMF/CNJ, seguido do ano de produção, ex: "Relatório UMF/CNJ 2021"
        2. Um resumo conciso do conteúdo em até 100 palavras

        Texto do documento:
        {texto_inicial}
        
        Responda apenas com o título e resumo separados por uma linha em branco.
        """

        response = llm.invoke(prompt)

        # Processar a resposta
        partes = response.content.strip().split("\n\n", 1)
        if len(partes) >= 2:
            titulo, resumo = str(partes[0].strip()), str(partes[1].strip())
        else:
            titulo = str(partes[0].strip())
            resumo = "Resumo não disponível."

        # Se o título estiver vazio, use o nome do arquivo
        if not titulo:
            titulo = str(file_name).replace(".pdf", "")

        return titulo, resumo

    except Exception as e:
        st.toast(f"Erro ao extrair metadados: {str(e)}", icon="⚠️")
        return (
            str(file_name).replace(".pdf", ""),
            "Não foi possível gerar um resumo deste documento.",
        )


def process_uploaded_file(uploaded_file):
    """Processa um arquivo PDF carregado."""
    # Verifica se o arquivo já foi processado
    if uploaded_file.name in st.session_state.processed_files:
        st.toast(f"O arquivo {uploaded_file.name} já foi processado.", icon="⚠️")
        return False

    try:
        with st.spinner(f"Processando {uploaded_file.name}..."):
            # Salva o PDF na pasta de armazenamento
            file_path = save_uploaded_pdf(uploaded_file)

            # Processa o PDF
            documents = process_pdf(file_path)

            if not documents:
                st.toast(
                    f"Nenhum conteúdo extraído do arquivo {uploaded_file.name}.",
                    icon="⚠️",
                )
                return False

            # Inicializa o modelo LLM para uso em metadados
            llm = initialize_llm()

            # Extrair todo o texto do documento para análise
            document_text = " ".join([doc.page_content for doc in documents])

            # Gerar título e resumo para o documento
            try:
                title, summary = extract_document_metadata(
                    document_text, uploaded_file.name, llm
                )
            except Exception as e:
                st.toast(f"Erro ao extrair metadados: {str(e)}", icon="⚠️")
                # Fallback para valores mais simples
                title = uploaded_file.name.replace(".pdf", "")
                summary = "Resumo não disponível devido a um erro na extração."

            # Armazenar os metadados do documento
            st.session_state.document_metadata[uploaded_file.name] = {
                "title": title,
                "summary": summary,
                "file_path": file_path,
            }

            # Inicializa embeddings
            embeddings = get_embeddings()

            # Verifica se temos um problema com o tenant
            error_flag_path = os.path.join(VECTOR_DB_DIR, "tenant_error.flag")
            if (
                os.path.exists(error_flag_path)
                or st.session_state.tenant_error_detected
            ):
                # Se houver problemas, força limpeza antes de criar nova base
                force_clean_vectordb()
                # Limpa as flags
                if os.path.exists(error_flag_path):
                    os.remove(error_flag_path)
                st.session_state.tenant_error_detected = False
                st.session_state.vector_store = None

            # Cria ou atualiza o vector store
            try:
                if st.session_state.vector_store is None:
                    import chromadb

                    # Usar o novo cliente ChromaDB conforme documentação de migração
                    st.session_state.vector_store = create_vector_store(
                        documents=documents,
                        embeddings=embeddings,
                        persist_directory=VECTOR_DB_DIR,
                    )
                else:
                    st.session_state.vector_store.add_documents(documents)
                    # Trata a persistência considerando as diferentes versões da API do Chroma
                    try:
                        if hasattr(st.session_state.vector_store, "persist"):
                            st.session_state.vector_store.persist()
                    except Exception as e:
                        st.toast("Persistência automática ativada", icon="ℹ️")
            except Exception as e:
                error_str = str(e)
                if (
                    "tenant default_tenant" in error_str
                    and "Could not connect" in error_str
                ):
                    # Erro de tenant ao processar arquivo
                    st.session_state.tenant_error_detected = True
                    with open(error_flag_path, "w") as f:
                        f.write(f"Error detected during processing: {error_str}")

                    st.error(
                        "Erro ao conectar à base de dados. Tente reparar a base usando o botão 'Reparar Base de Dados'."
                    )
                    return False
                else:
                    raise  # Permite que outros erros sejam tratados normalmente

            # Atualiza a lista de arquivos processados
            st.session_state.processed_files.append(uploaded_file.name)

            # Inicializa o RAG chain se ainda não existir
            initialize_rag_chain(st.session_state.vector_store)

        st.toast(f"Documento '{title}' processado com sucesso!", icon="✅")
        return True

    except Exception as e:
        st.toast(f"Erro ao processar o arquivo: {str(e)}", icon="❌")
        st.error(f"Erro ao processar o arquivo: {str(e)}")
        return False


def reset_chat():
    """Reseta o histórico de chat."""
    if "chat_messages" in st.session_state:
        st.session_state.chat_messages = []
        st.toast("Histórico de conversa limpo!", icon="🗑️")
    return True


def clear_all_data():
    """Limpa todos os dados do vectordb e PDFs armazenados."""
    try:
        # Primeiro, libere a referência do vector_store na sessão
        # Isso é crucial para liberar os arquivos antes de tentar excluí-los
        if st.session_state.vector_store is not None:
            # Tenta fechar a conexão com o Chroma explicitamente
            try:
                if hasattr(st.session_state.vector_store, "_client"):
                    st.session_state.vector_store._client.close()
                elif hasattr(st.session_state.vector_store, "_collection"):
                    if hasattr(st.session_state.vector_store._collection, "_client"):
                        st.session_state.vector_store._collection._client.close()
            except Exception as e:
                # Ignora erros ao tentar fechar explicitamente
                print(f"Aviso: Não foi possível fechar a conexão com o Chroma: {e}")

            # Remove a referência
            st.session_state.vector_store = None

            # Força a coleta de lixo para liberar recursos
            import gc

            gc.collect()

        # Aguarda um pouco para garantir que os recursos sejam liberados
        import time

        time.sleep(1)

        # Agora tenta limpar os diretórios
        if os.path.exists(VECTOR_DB_DIR):
            try:
                # Tenta excluir o diretório inteiro primeiro
                shutil.rmtree(VECTOR_DB_DIR)
            except OSError as e:
                # Se falhar, tenta excluir arquivo por arquivo
                st.toast(f"Tentando limpeza alternativa do vectordb...", icon="ℹ️")
                for root, dirs, files in os.walk(VECTOR_DB_DIR, topdown=False):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                        except OSError:
                            # Ignora arquivos que não podem ser excluídos
                            pass
                    for dir in dirs:
                        try:
                            dir_path = os.path.join(root, dir)
                            os.rmdir(dir_path)
                        except OSError:
                            # Ignora diretórios que não podem ser excluídos
                            pass

                # Por fim, tenta remover o diretório raiz
                try:
                    os.rmdir(VECTOR_DB_DIR)
                except OSError:
                    pass

            # Recria o diretório
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)

        # Remove os PDFs armazenados
        if os.path.exists(PDF_STORAGE_DIR):
            for file in os.listdir(PDF_STORAGE_DIR):
                if file.endswith(".pdf"):
                    try:
                        os.remove(os.path.join(PDF_STORAGE_DIR, file))
                    except OSError as e:
                        st.toast(
                            f"Não foi possível excluir o arquivo {file}: {str(e)}",
                            icon="⚠️",
                        )

        # Reseta as variáveis de sessão
        st.session_state.processed_files = []
        st.session_state.document_metadata = {}
        st.session_state.rag_chain = None

        # Limpa o histórico de chat
        reset_chat()

        st.toast(
            "Dados limpos com sucesso! Recarregue a página para uma limpeza completa.",
            icon="✅",
        )
        st.balloons()  # Celebra o sucesso com balões

        # Flag para indicar que a limpeza foi bem-sucedida, usada para mostrar o botão de recarregamento
        st.session_state.data_cleared = True

        return True
    except Exception as e:
        st.toast(f"Erro ao limpar dados: {str(e)}", icon="❌")
        st.error(f"Erro ao limpar dados: {str(e)}")
        st.info(
            "Tente recarregar a página e limpar os dados novamente, ou reinicie o servidor Streamlit."
        )
        return False


def sanitize_for_html(text):
    """Sanitiza o texto para uso seguro em atributos HTML."""
    if not text:
        return ""
    # Substitui aspas duplas e outros caracteres problemáticos
    # Não remove quebras de linha, apenas sanitiza caracteres especiais
    return (
        str(text)
        .replace('"', "&quot;")
        .replace("'", "&#39;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def main():
    with st.spinner("Inicializando..."):
        # Initialize vector store and RAG chain
        vector_store = initialize_vector_store()
        rag_chain = initialize_rag_chain(vector_store) if vector_store else None

    # Título
    st.title("📚 RAG UMF/CNJ - Relatórios Anuais")
    st.subheader("Consulta das atividades realizadas pela UMF/CNJ")

    # Adiciona a mensagem informativa para o usuário
    st.info(
        """
    Formule perguntas sobre o funcionamento e as atividades da Unidade de Monitoramento e Fiscalização do Sistema Interamericano de Direitos Humanos (UMF/CNJ). Aqui estão alguns exemplos de perguntas que você pode fazer:

    - Como a UMF/CNJ coleta dados para monitorar o seu plano de trabalho?
    - Quais são as ações sugeridas pela UMF/CNJ quando não há condenações específicas na jurisdição do tribunal?
    - Que tipos de capacitação foram propostos para profissionais que lidam com saúde mental no caso Ximenes Lopes vs. Brasil?
    - Como a UMF/CNJ utiliza cursos de capacitação para lidar com problemas locais, como conflitos fundiários no Paraná?
    - Pode fornecer informações sobre como a UMF/CNJ avalia a qualidade das perícias criminais no Brasil?
    """
    )

    # Tentar limpar diretórios de vectordb órfãos durante a inicialização
    if cleanup_orphaned_vectordb():
        st.toast("Dados antigos foram limpos automaticamente", icon="🧹")

    # Verificar se os dados foram recentemente limpos
    if st.session_state.data_cleared:
        st.success(
            "Dados limpos com sucesso! É recomendado recarregar a página para garantir que todos os recursos sejam liberados corretamente."
        )

        if st.button("🔄 Recarregar Aplicação", type="primary"):
            # Use JavaScript para recarregar a página
            st.markdown(
                """
                <script>
                    window.parent.location.reload();
                </script>
                """,
                unsafe_allow_html=True,
            )
            # Fallback mensagem caso o JavaScript não funcione
            st.info(
                "Se a página não recarregar automaticamente, por favor recarregue manualmente usando o navegador."
            )

    # Verifica se há um erro de tenant detectado e mostra opção de reparação
    error_flag_path = os.path.join(VECTOR_DB_DIR, "tenant_error.flag")
    if os.path.exists(error_flag_path) or st.session_state.tenant_error_detected:
        st.error("⚠️ **Erro de conexão com o banco de dados vetorial detectado**")
        st.warning(
            "Um problema foi detectado na base de dados vetorial. "
            + "Isso pode ocorrer devido a uma desconexão abrupta ou problemas com o ChromaDB."
        )

        repair_col1, repair_col2 = st.columns([2, 1])
        with repair_col1:
            st.info(
                "A reparação irá limpar a base de dados existente. Você precisará recarregar seus documentos."
            )
        with repair_col2:
            if st.button("🔧 Reparar Base de Dados", type="primary"):
                # Limpa a base de dados
                force_clean_vectordb()

                # Remove a flag de erro
                if os.path.exists(error_flag_path):
                    os.remove(error_flag_path)

                # Reseta as flags de erro
                st.session_state.tenant_error_detected = False
                st.session_state.vector_store = None

                # Notifica o usuário
                st.toast("Base de dados reparada com sucesso!", icon="✅")
                st.success(
                    "Base de dados reparada. A página será recarregada para aplicar as mudanças."
                )
                st.markdown(
                    """
                    <script>
                        setTimeout(function() {
                            window.parent.location.reload();
                        }, 2000);
                    </script>
                    """,
                    unsafe_allow_html=True,
                )
                st.stop()

    # Inicialização do histórico de chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Inicializa ou carrega a base de vetores persistente
    initialize_vector_store()

    # Inicializa a cadeia RAG se necessário
    if st.session_state.vector_store:
        initialize_rag_chain(st.session_state.vector_store)

    # Barra lateral
    with st.sidebar:
        # st.header("📄 Carregar Documentos")

        # # Upload de arquivos
        # uploaded_files = st.file_uploader(
        #     "Selecione os PDFs da UMF/CNJ", type="pdf", accept_multiple_files=True
        # )

        # # Botão para processar os arquivos
        # if uploaded_files:
        #     if st.button("Processar Documentos"):
        #         for uploaded_file in uploaded_files:
        #             process_uploaded_file(uploaded_file)

        # Lista de arquivos processados
        if st.session_state.processed_files:
            # st.write("---")
            st.subheader("📑 Documentos Processados")

            # Alteramos a abordagem para usar componentes nativos do Streamlit
            for filename in st.session_state.processed_files:
                # Obter os metadados do documento
                metadata = st.session_state.document_metadata.get(filename, {})
                title = metadata.get("title", filename.replace(".pdf", ""))
                summary = metadata.get("summary", "Resumo não disponível.")

                # Garantir que o resumo seja uma string simples
                summary = str(summary)

                # Usando expander para mostrar informações do documento - mais confiável que tooltips
                with st.expander(f"📄 **{title}**"):
                    st.write(summary)
                    st.caption(f"Arquivo: {filename}")

        # Botão para limpar histórico
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Limpar Histórico"):
                reset_chat()

        # with col2:
        #     if st.button("Limpar Tudo", type="primary", use_container_width=True):
        #         clear_all_data()

    # Área principal
    if not st.session_state.processed_files:
        st.info("👈 Carregue os documentos da UMF/CNJ no painel lateral para começar.")
        st.stop()

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
                if st.session_state.rag_chain is None:
                    # Reinicializa a cadeia RAG se estiver ausente
                    initialize_rag_chain(st.session_state.vector_store)

                # Recupera os chunks relevantes
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": 15}
                )
                docs = retriever.get_relevant_documents(prompt)

                # Mostra os chunks recuperados em um expander
                with st.expander("🔍 Ver chunks recuperados"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content)
                        st.markdown("---")

                # Gera a resposta
                response = st.session_state.rag_chain.invoke({"question": prompt})
                st.write(response)

        # Adiciona ao histórico
        chat_history.add_ai_message(response)


if __name__ == "__main__":
    main()
