"""
Módulo para gerenciar a memória e o histórico de conversas.
"""

from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class StreamlitChatHistory(BaseChatMessageHistory):
    """
    Implementação de historico de chat compatível com Streamlit.
    Usa session_state para armazenar mensagens entre interações.
    """

    def __init__(self, session_state, key: str = "chat_messages"):
        """
        Inicializa o histórico de chat do Streamlit.

        Args:
            session_state: O objeto session_state do Streamlit.
            key: A chave para armazenar as mensagens no session_state.
        """
        self.session_state = session_state
        self.key = key

        # Inicializa a lista de mensagens se não existir
        if self.key not in self.session_state:
            self.session_state[self.key] = []

    @property
    def messages(self) -> List[BaseMessage]:
        """Retorna a lista de mensagens."""
        # Verifica se a chave existe e inicializa se necessário
        if self.key not in self.session_state:
            self.session_state[self.key] = []
        return self.session_state[self.key]

    def add_message(self, message: BaseMessage) -> None:
        """
        Adiciona uma mensagem ao histórico.

        Args:
            message: A mensagem a ser adicionada.
        """
        # Verifica se a chave existe e inicializa se necessário
        if self.key not in self.session_state:
            self.session_state[self.key] = []
        self.session_state[self.key].append(message)

    def add_user_message(self, message: str) -> None:
        """
        Adiciona uma mensagem do usuário ao histórico.

        Args:
            message: O conteúdo da mensagem do usuário.
        """
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """
        Adiciona uma mensagem do assistente ao histórico.

        Args:
            message: O conteúdo da mensagem do assistente.
        """
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """Limpa todas as mensagens do histórico."""
        self.session_state[self.key] = []


def get_conversation_memory(
    chat_history: StreamlitChatHistory,
) -> ConversationBufferMemory:
    """
    Cria uma memória de conversa a partir do histórico de chat.

    Args:
        chat_history: O histórico de chat a ser usado.

    Returns:
        Uma instância de ConversationBufferMemory.
    """
    # Inicializa uma nova memória de conversa
    try:
        memory = ConversationBufferMemory(
            chat_memory=chat_history, return_messages=True, memory_key="chat_history"
        )
        return memory
    except Exception as e:
        # Em caso de erro, cria uma nova memória com chat_memory vazio
        print(f"Erro ao criar memória de conversa: {e}")
        chat_history.clear()  # Limpa o histórico para começar novamente
        memory = ConversationBufferMemory(
            chat_memory=chat_history, return_messages=True, memory_key="chat_history"
        )
        return memory
