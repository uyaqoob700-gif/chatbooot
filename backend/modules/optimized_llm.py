from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def create_pec_specialized_prompt() -> PromptTemplate:
    """Create a specialized prompt for PEC engineering questions."""
    
    prompt_template = """You are a helpful AI assistant for Pakistan Engineering Council (PEC) information. Answer questions about PEC regulations, procedures, and policies in a conversational, helpful manner.

CONTEXT FROM PEC DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided PEC context above
2. If the information is not in the context, simply say: "I don't have that specific information in the uploaded PEC documents."
3. Be conversational and helpful - like talking to a colleague
4. Provide direct, practical answers without unnecessary structure
5. If you have partial information, share what you know
6. Use simple language appropriate for engineers
7. Keep responses concise but complete
8. If mentioning specific requirements, fees, or procedures, cite them exactly as they appear

ANSWER:"""

    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )


def create_query_expansion_prompt(original_query: str) -> str:
    """Expand the query with related terms for better retrieval."""
    
    # PEC-specific query expansion
    expansion_map = {
        'registration': ['registration', 'register', 'enrollment', 'enroll', 'apply', 'application'],
        'engineer': ['engineer', 'engineering', 'professional engineer', 'pe', 'registered engineer', 're'],
        'firm': ['firm', 'company', 'consultant', 'contractor', 'organization', 'enterprise'],
        'examination': ['examination', 'exam', 'test', 'epe', 'engineering practice examination', 'assessment'],
        'cpd': ['cpd', 'continuing professional development', 'training', 'course', 'education', 'learning'],
        'fee': ['fee', 'fees', 'payment', 'cost', 'charge', 'amount'],
        'renewal': ['renewal', 'renew', 'expiry', 'expire', 'validity', 'valid'],
        'regulation': ['regulation', 'regulations', 'act', 'rule', 'rules', 'policy', 'policies']
    }
    
    expanded_terms = [original_query]
    query_lower = original_query.lower()
    
    for key, terms in expansion_map.items():
        if key in query_lower:
            expanded_terms.extend(terms)
    
    # Remove duplicates and return as space-separated string
    return ' '.join(list(set(expanded_terms)))


def get_optimized_llm_chain(retriever: BaseRetriever, temperature: float = 0.1):
    """
    Create optimized LLM chain with PEC-specialized prompt and lower temperature.
    
    Args:
        retriever: Document retriever
        temperature: LLM temperature (0-1, lower = more focused)
        
    Returns:
        RetrievalQA chain
    """
    try:
        # Initialize Groq LLM with optimized settings
        llm = ChatGroq(
            api_key=os.environ["GROQ_API_KEY"],
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,  # Lower temperature for more focused answers
            max_tokens=3072,  # Increased for more comprehensive answers
            timeout=45,  # Increased timeout
            max_retries=3
        )
        
        logger.info(f"Initialized optimized Groq LLM: {os.getenv('GROQ_MODEL')} (temp: {temperature})")
        
        # Create PEC-specialized prompt
        prompt = create_pec_specialized_prompt()
        
        # Create RetrievalQA chain with optimized settings
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            verbose=False  # Set to True for debugging
        )
        
        logger.info("Successfully created optimized LLM chain")
        return chain
        
    except Exception as e:
        logger.error(f"Error creating optimized LLM chain: {e}")
        raise


def get_standard_llm_chain(retriever: BaseRetriever, temperature: float = 0.3):
    """
    Create standard LLM chain (fallback).
    """
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            api_key=os.environ["GROQ_API_KEY"],
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
            max_tokens=2048,
            timeout=30,
            max_retries=2
        )
        
        logger.info(f"Initialized standard Groq LLM: {os.getenv('GROQ_MODEL')}")
        
        # Create standard prompt template
        prompt_template = """You are a helpful AI assistant answering questions based on the provided context from uploaded documents.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context above
- If the answer is not in the context, say "I don't have enough information in the uploaded documents to answer that question."
- Be concise but complete
- Use bullet points for lists when appropriate
- Cite specific information from the context when relevant

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("Successfully created standard LLM chain")
        return chain
        
    except Exception as e:
        logger.error(f"Error creating standard LLM chain: {e}")
        raise
