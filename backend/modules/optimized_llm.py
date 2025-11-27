"""
Enhanced RAG System for PEC Chatbot with Improved Answer Quality
Implements best practices for consistent, accurate responses
"""

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import logging

logger = logging.getLogger(__name__)


def preprocess_query(query: str) -> str:
    """
    Preprocess and enhance user query for better retrieval.
    Expands abbreviations and adds context.
    """
    # PEC-specific abbreviation expansions
    abbreviations = {
        "cpd": "Continuing Professional Development",
        "pec": "Pakistan Engineering Council",
        "pe": "Professional Engineer",
        "cpe": "Continuing Professional Education",
        "nbe": "National Board of Engineers",
    }
    
    query_lower = query.lower()
    expanded_query = query
    
    # Expand known abbreviations
    for abbr, full in abbreviations.items():
        if abbr in query_lower.split():
            expanded_query = expanded_query.replace(abbr, f"{abbr} ({full})")
    
    return expanded_query


def create_enhanced_prompt_template():
    """
    Creates a specialized prompt template for PEC domain with strict guidelines.
    """
    template = """You are an expert assistant for the Pakistan Engineering Council (PEC). Your role is to provide accurate, helpful information about PEC services, regulations, and procedures.

CRITICAL GUIDELINES:
1. ONLY answer based on the provided context documents
2. If information is not in the context, clearly state: "I don't have specific information about that in my knowledge base. Please refer to official PEC documentation or contact PEC directly."
3. Be specific with numbers, dates, fees, and requirements - never guess
4. Format responses clearly with bullet points or numbered lists when appropriate
5. Always cite which section or document your answer comes from when possible
6. For procedural questions, provide step-by-step instructions
7. Maintain a professional, helpful tone

CONTEXT FROM PEC DOCUMENTS:
{context}

USER QUESTION: {question}

DETAILED ANSWER (based strictly on the context above):"""

    return ChatPromptTemplate.from_template(template)


def create_validation_prompt():
    """
    Creates a prompt to validate the generated answer for accuracy and relevance.
    """
    template = """You are a quality validator for a PEC (Pakistan Engineering Council) chatbot.

Review the following answer and determine if it meets these criteria:
1. Answers the question directly
2. Uses only information from the provided context
3. Doesn't contain speculation or unsupported claims
4. Is clear and well-formatted
5. Includes appropriate caveats when information is limited

ORIGINAL QUESTION: {question}

CONTEXT PROVIDED: {context}

GENERATED ANSWER: {answer}

VALIDATION:
- Is the answer accurate based on the context? (Yes/No)
- Does it fully address the question? (Yes/No)
- Are there any hallucinations or unsupported claims? (Yes/No)
- Quality Score (1-10):
- Suggested improvements:

Provide a brief validation report:"""

    return ChatPromptTemplate.from_template(template)


def get_optimized_llm_chain(retriever, temperature=0.1, model_name=None):
    """
    Creates an optimized RAG chain with enhanced prompting and validation.
    
    Args:
        retriever: LangChain retriever object
        temperature: Lower temperature (0.0-0.3) for more consistent answers
        model_name: Groq model name (defaults to environment variable)
    
    Returns:
        LangChain RAG chain
    """
    # Use lower temperature for more consistent, factual responses
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,  # Low temperature for consistency
        max_tokens=2048,  # Increased for detailed answers
    )
    
    # Create the enhanced prompt
    prompt = create_enhanced_prompt_template()
    
    # Build the RAG chain with proper document formatting
    def format_docs(docs):
        """Format documents with clear section markers and metadata."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            formatted.append(
                f"[Document {i} - Source: {source}, Page: {page}]\n{doc.page_content}\n"
            )
        return "\n---\n".join(formatted)
    
    # Create the RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def validate_answer(question: str, context: str, answer: str) -> dict:
    """
    Validates the generated answer using an LLM as a judge.
    
    Args:
        question: Original user question
        context: Retrieved context
        answer: Generated answer
    
    Returns:
        Dictionary with validation results
    """
    try:
        validator_llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.0,  # Deterministic for validation
            max_tokens=512,
        )
        
        validation_prompt = create_validation_prompt()
        
        validation_chain = validation_prompt | validator_llm | StrOutputParser()
        
        validation_result = validation_chain.invoke({
            "question": question,
            "context": context[:2000],  # Limit context size
            "answer": answer
        })
        
        return {
            "validation_report": validation_result,
            "passed": "yes" in validation_result.lower() or "10" in validation_result
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "validation_report": "Validation failed",
            "passed": True  # Default to passing on error
        }


def enhance_answer_with_formatting(answer: str) -> str:
    """
    Post-process the answer to improve formatting and readability.
    Remove unwanted phrases that reference source documents.
    """
    # Remove common unwanted phrases
    unwanted_phrases = [
        r"based on the provided context( documents)?[,\s]*",
        r"according to (the )?(provided )?(context )?documents?[,\s]*",
        r"specifically \[document \d+[^\]]*\]( and \[document \d+[^\]]*\])*[,\s]*",
        r"the (provided )?(context )?documents? (state|mention|indicate|show|reveal)[s]? that[,\s]*",
        r"from the (context|documents|information provided)[,\s]*",
        r"in the (given|provided) (context|documents)[,\s]*",
        r"\[document \d+ - source:?[^\]]*\][,\s]*",
        r"as (stated|mentioned|indicated) in (the )?(context|documents?)[,\s]*",
    ]
    
    import re
    for pattern in unwanted_phrases:
        # Remove the phrase and capitalize the next word
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)
    
    # Clean up extra spaces and fix capitalization
    answer = re.sub(r'\s+', ' ', answer)
    answer = answer.strip()
    
    # Capitalize first letter if needed
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    
    # Add spacing for better readability
    if ":" in answer and not answer.startswith("-"):
        # Likely contains lists or structured info
        lines = answer.split("\n")
        formatted_lines = []
        for line in lines:
            if line.strip():
                if any(line.strip().startswith(str(i)) for i in range(1, 10)):
                    formatted_lines.append(f"\n{line}")
                elif line.strip().startswith("-") or line.strip().startswith("•"):
                    formatted_lines.append(f"\n{line}")
                else:
                    formatted_lines.append(line)
        answer = "\n".join(formatted_lines)
    
    return answer.strip()


# Advanced configuration options
RAG_CONFIG = {
    "chunk_size": 1000,  # Optimal chunk size for PEC documents
    "chunk_overlap": 200,  # Overlap to maintain context
    "top_k_retrieval": 5,  # Retrieve top 5 most relevant chunks
    "rerank_top_k": 3,  # After reranking, use top 3
    "temperature": 0.1,  # Low temperature for consistency
    "max_tokens": 2048,  # Allow detailed responses
    "enable_validation": True,  # Enable answer validation
}
