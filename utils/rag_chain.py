"""
Módulo para implementação da cadeia RAG.
"""

from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Template de prompt para o RAG
RAG_PROMPT_TEMPLATE = """
Você é um assistente especializado em direitos humanos e no Sistema Interamericano de Direitos Humanos.
Sua função é responder perguntas sobre os documentos da Unidade de Monitoramento e Fiscalização do Sistema Interamericano de Direitos Humanos (UMF/CNJ).

Utilize apenas o contexto fornecido abaixo para responder à pergunta. Se a informação não estiver no contexto, 
diga que não possui essa informação e sugira que o usuário reformule a pergunta ou consulte os documentos originais 
da UMF/CNJ.

Ao citar informações, mencione de qual documento elas foram extraídas.

Histórico da conversa:
{chat_history}

Contexto:
{context}

Pergunta: {question}

Resposta:
"""


def format_docs(docs):
    """
    Formata uma lista de documentos em um único texto.

    Args:
        docs: Lista de documentos a serem formatados.

    Returns:
        String com o conteúdo formatado dos documentos.
    """
    return "\n\n".join(
        f"DOCUMENTO [{doc.metadata.get('source', 'Desconhecido')}]:\n"
        + doc.page_content
        for doc in docs
    )


def create_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    memory: Optional[ConversationBufferMemory] = None,
):
    """
    Cria uma cadeia RAG completa.

    Args:
        retriever: Recuperador para buscar documentos relevantes.
        llm: Modelo de linguagem para gerar respostas.
        memory: Memória de conversa para manter contexto (opcional).

    Returns:
        A cadeia RAG configurada.
    """
    # Cria o template de prompt
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # Define a preparação dos inputs
    if memory:

        def _get_chat_history(inputs):
            return memory.chat_memory.messages

        # Preparação com memória
        prepare_inputs = {
            "context": lambda inputs: format_docs(
                retriever.get_relevant_documents(inputs["question"])
            ),
            "question": lambda inputs: inputs["question"],
            "chat_history": _get_chat_history,
        }
    else:
        # Preparação sem memória
        prepare_inputs = {
            "context": lambda inputs: format_docs(
                retriever.get_relevant_documents(inputs["question"])
            ),
            "question": lambda inputs: inputs["question"],
            "chat_history": lambda _: [],
        }

    # Constrói a cadeia RAG
    rag_chain = RunnableParallel(prepare_inputs) | prompt | llm | StrOutputParser()

    return rag_chain
