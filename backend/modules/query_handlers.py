from langchain.chains import RetrievalQA
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def query_chain(chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """
    Query the RAG chain and format the response.
    
    Args:
        chain: RetrievalQA chain
        question: User question
        
    Returns:
        Dictionary with answer and sources
    """
    try:
        logger.info(f"Querying chain with question: {question}")
        
        # Query the chain
        response = chain({"query": question})
        
        # Extract answer
        answer = response.get("result", "")
        
        # Extract and format sources
        sources = []
        source_docs = response.get("source_documents", [])
        
        for i, doc in enumerate(source_docs[:3]):  # Limit to top 3 sources
            source_info = {
                "index": i + 1,
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        logger.info(f"Successfully generated answer with {len(source_docs)} sources")
        
        return {
            "answer": answer,
            "sources": sources,
            "source_count": len(source_docs)
        }
        
    except Exception as e:
        logger.error(f"Error querying chain: {e}")
        raise