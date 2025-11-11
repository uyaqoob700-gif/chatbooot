from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
import os
import logging

logger = logging.getLogger(__name__)


def get_llm_chain(retriever: BaseRetriever, temperature: float = 0.3):
    """
    Create LLM chain with Groq API.
    
    Args:
        retriever: Document retriever
        temperature: LLM temperature (0-1)
        
    Returns:
        RetrievalQA chain
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
        
        logger.info(f"Initialized Groq LLM: {os.getenv('GROQ_MODEL')}")
        
        # Create custom prompt template
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
        
        logger.info("Successfully created LLM chain")
        return chain
        
    except Exception as e:
        logger.error(f"Error creating LLM chain: {e}")
        raise