import os
import logging
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


IntentLabel = Literal["smalltalk", "capability", "rag", "general"]


def classify_query_with_groq(question: str) -> IntentLabel:
    """
    Use Groq LLM to classify the user query into one of:
    - smalltalk: greetings and chit-chat
    - capability: ask about what the bot can do
    - rag: requires looking up uploaded documents
    - general: can be answered without retrieval
    """
    try:
        llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.0,
            max_tokens=64
        )
        system = (
            "You are a classifier. Read the user's question and output exactly one label "
            "from this set: smalltalk, capability, rag, general. "
            "- smalltalk: greetings, how are you, thanks, goodbye.\n"
            "- capability: asking what the assistant can do or how it helps.\n"
            "- rag: requires specific facts from the user's uploaded documents (PEC rules, fees, forms, deadlines, procedures).\n"
            "- general: generic PEC or domain knowledge answerable without documents.\n"
            "Output only the label, nothing else."
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=question)
        ]
        label = llm.invoke(messages).content.strip().lower()
        if label in {"smalltalk", "capability", "rag", "general"}:
            return label  # type: ignore[return-value]
        logger.warning(f"Unexpected intent label '{label}', defaulting to 'rag'")
        return "rag"  # type: ignore[return-value]
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}. Defaulting to 'rag'")
        return "rag"  # type: ignore[return-value]





