"""
Enhanced intent classification for better query routing
"""

import os
import logging
from typing import Literal

logger = logging.getLogger(__name__)

IntentType = Literal["rag", "general", "greeting", "capability"]


def classify_query_with_groq(query: str) -> IntentType:
    """
    Classify user query intent using Groq LLM for accurate routing.
    
    Categories:
    - rag: Questions about PEC services, procedures, requirements (needs document retrieval)
    - general: General questions about PEC that can be answered without documents
    - greeting: Simple greetings and conversation starters
    - capability: Questions about what the bot can do
    
    Args:
        query: User's question
    
    Returns:
        Intent classification
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.0,  # Deterministic for classification
            max_tokens=50
        )
        
        system_prompt = """You are an intent classifier for a Pakistan Engineering Council (PEC) assistant chatbot.

Classify the user's query into ONE of these categories:

1. "rag" - Questions that need specific information from PEC documents:
   - Registration requirements, fees, documents needed
   - Licensing procedures and timelines
   - CPD (Continuing Professional Development) details
   - Project approval processes
   - Specific rules, regulations, or policies
   - Application procedures
   - Eligibility criteria
   Examples: "What documents are needed for engineer registration?", "What is the CPD requirement?", "How to apply for firm license?"

2. "general" - General questions about PEC that don't need document lookup:
   - What PEC does (overview)
   - General information about engineering in Pakistan
   - Broad questions about PEC's role
   Examples: "What is PEC?", "Why is PEC important?"

3. "greeting" - Simple greetings and small talk:
   - Hi, hello, hey, good morning, etc.
   - How are you?
   - Thank you
   Examples: "Hello", "Hi there", "Good morning"

4. "capability" - Questions about bot capabilities:
   - What can you do?
   - How can you help?
   - What services do you provide?
   Examples: "What can you help me with?", "How do you work?"

Respond with ONLY ONE WORD: rag, general, greeting, or capability"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Classify this query: '{query}'")
        ]
        
        response = llm.invoke(messages).content.strip().lower()
        
        # Validate response
        valid_intents = {"rag", "general", "greeting", "capability"}
        if response in valid_intents:
            logger.info(f"Query classified as: {response}")
            return response
        else:
            # Default to rag if unclear
            logger.warning(f"Unclear classification '{response}', defaulting to 'rag'")
            return "rag"
            
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        # Default to rag for safety (will search documents)
        return "rag"


def get_predefined_response(intent: IntentType) -> str:
    """
    Get predefined responses for non-RAG intents.
    """
    responses = {
        "capability": """I can help you with information about PEC (Pakistan Engineering Council) services including:

• **Engineer Registration** - Requirements, documents, fees, and application process
• **Firm Licensing** - How to register your engineering firm
• **CPD (Continuing Professional Development)** - Requirements and compliance
• **Project Approvals** - Procedures and timelines
• **Regulations & Policies** - Current PEC rules and guidelines

Ask me specific questions about any of these topics, such as:
- "What documents are needed for engineer registration?"
- "What is the fee for firm licensing?"
- "How many CPD hours are required?"
- "What is the process for project approval?"

I'm here to help! 😊""",
        
        "greeting": """Hello! 👋 I'm your PEC Assistant. 

I'm here to help you with information about:
- Engineer Registration
- Firm Licensing
- CPD Requirements
- Project Approvals
- PEC Regulations

What would you like to know about PEC services today?"""
    }
    
    return responses.get(intent, "")


def should_use_rag(query: str) -> bool:
    """
    Quick heuristic check to determine if RAG is needed.
    Useful as a fast pre-filter before LLM classification.
    
    Returns:
        True if query likely needs document retrieval
    """
    # Keywords that typically indicate need for specific information
    rag_indicators = [
        # Question words for specifics
        "what", "how", "when", "where", "which", "who",
        
        # PEC-specific terms
        "registration", "license", "fee", "requirement", "document",
        "eligibility", "criteria", "procedure", "process", "application",
        "cpd", "hours", "certificate", "renewal", "approval",
        "form", "timeline", "deadline", "cost", "payment",
        
        # Action words
        "apply", "register", "submit", "need", "required",
        "obtain", "get", "acquire", "renew"
    ]
    
    query_lower = query.lower()
    
    # Check if query contains RAG indicators
    has_indicators = any(indicator in query_lower for indicator in rag_indicators)
    
    # Check if it's a specific question (contains question mark or question word)
    is_question = "?" in query or any(query_lower.startswith(qw) for qw in ["what", "how", "when", "where", "which", "who", "why"])
    
    return has_indicators or is_question


def enhance_query_for_retrieval(query: str) -> str:
    """
    Enhance query with additional context for better retrieval.
    """
    # Add "PEC" context if not present
    if "pec" not in query.lower():
        query = f"PEC Pakistan Engineering Council {query}"
    
    # Expand common abbreviations
    expansions = {
        "reg": "registration",
        "lic": "license",
        "req": "requirement",
        "doc": "document",
        "app": "application"
    }
    
    words = query.split()
    enhanced_words = []
    
    for word in words:
        word_lower = word.lower().rstrip("s.,!?")
        if word_lower in expansions:
            enhanced_words.append(f"{word} {expansions[word_lower]}")
        else:
            enhanced_words.append(word)
    
    return " ".join(enhanced_words)
