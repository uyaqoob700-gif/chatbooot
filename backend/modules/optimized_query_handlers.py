"""
Advanced query handler with hybrid search, reranking, and answer validation
"""

import logging
from typing import List, Dict, Any
import re
from datetime import datetime

logger = logging.getLogger(__name__)


def extract_keywords(query: str) -> List[str]:
    """
    Extract important keywords from query for hybrid search.
    """
    # PEC-specific important terms
    important_terms = [
        "registration", "license", "fee", "requirement", "document",
        "eligibility", "cpd", "professional", "engineer", "firm",
        "project", "approval", "application", "certificate", "renewal"
    ]
    
    words = query.lower().split()
    keywords = [w for w in words if w in important_terms or len(w) > 4]
    
    return keywords[:5]  # Top 5 keywords


def hybrid_search(query: str, embed_model, index, top_k=10) -> List[Dict]:
    """
    Perform hybrid search combining semantic and keyword search.
    
    Args:
        query: User query
        embed_model: HuggingFace embeddings model
        index: Pinecone index
        top_k: Number of results to retrieve
    
    Returns:
        List of retrieved documents with scores
    """
    try:
        # 1. Semantic search using embeddings
        query_embedding = embed_model.embed_query(query)
        
        # 2. Extract keywords for metadata filtering
        keywords = extract_keywords(query)
        
        # 3. Search with both semantic similarity and keyword boost
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # 4. Boost scores for keyword matches in metadata
        enhanced_results = []
        for match in results.matches:
            score = match.score
            text = match.metadata.get('text', '').lower()
            
            # Boost score if keywords are found
            keyword_boost = sum(1 for kw in keywords if kw in text) * 0.1
            enhanced_score = min(score + keyword_boost, 1.0)
            
            enhanced_results.append({
                'id': match.id,
                'score': enhanced_score,
                'text': match.metadata.get('text', ''),
                'source': match.metadata.get('source', 'Unknown'),
                'page': match.metadata.get('page', 'N/A'),
            })
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Hybrid search retrieved {len(enhanced_results)} documents")
        return enhanced_results
        
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        return []


def rerank_results(query: str, results: List[Dict], top_k=3) -> List[Dict]:
    """
    Rerank results using simple but effective heuristics.
    
    Args:
        query: User query
        results: Retrieved documents
        top_k: Number of top results to return
    
    Returns:
        Reranked list of documents
    """
    try:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for result in results:
            text_lower = result['text'].lower()
            text_words = set(text_lower.split())
            
            # Calculate additional relevance signals
            
            # 1. Exact phrase match
            exact_match_bonus = 0.2 if query_lower in text_lower else 0
            
            # 2. Word overlap (Jaccard similarity)
            overlap = len(query_words & text_words) / len(query_words | text_words)
            
            # 3. Position of query terms (earlier is better)
            position_bonus = 0
            for word in query_words:
                if word in text_lower:
                    pos = text_lower.index(word)
                    position_bonus += (1 - pos / len(text_lower)) * 0.1
            
            # 4. Document completeness (prefer longer, more detailed docs)
            completeness_bonus = min(len(result['text']) / 1000, 0.1)
            
            # Combined rerank score
            rerank_score = (
                result['score'] * 0.5 +  # Original semantic score
                exact_match_bonus +
                overlap * 0.2 +
                position_bonus +
                completeness_bonus
            )
            
            result['rerank_score'] = rerank_score
        
        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Reranked to top {top_k} results")
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Reranking error: {e}")
        return results[:top_k]


def detect_answer_issues(answer: str, context: str, query: str) -> Dict[str, Any]:
    """
    Detect potential issues in generated answer.
    
    Returns:
        Dictionary with issue flags and confidence score
    """
    issues = []
    confidence = 1.0
    
    # 1. Check for hedge words indicating uncertainty
    hedge_words = ["might", "maybe", "possibly", "perhaps", "unclear", 
                   "not sure", "cannot confirm", "unable to find"]
    if any(word in answer.lower() for word in hedge_words):
        issues.append("Contains uncertainty indicators")
        confidence -= 0.2
    
    # 2. Check if answer is too short (likely incomplete)
    if len(answer.split()) < 20:
        issues.append("Answer may be too brief")
        confidence -= 0.1
    
    # 3. Check for generic responses
    generic_phrases = ["contact pec", "refer to official", "check the website"]
    if any(phrase in answer.lower() for phrase in generic_phrases):
        issues.append("Generic redirect response")
        confidence -= 0.15
    
    # 4. Check if key terms from query are addressed
    query_terms = set(query.lower().split())
    important_terms = [t for t in query_terms if len(t) > 4]
    addressed_terms = sum(1 for term in important_terms if term in answer.lower())
    
    if important_terms and addressed_terms / len(important_terms) < 0.5:
        issues.append("May not fully address the question")
        confidence -= 0.2
    
    # 5. Check for potential hallucination markers
    unsupported_claims = ["it is known that", "studies show", "experts say"]
    if any(claim in answer.lower() for claim in unsupported_claims):
        issues.append("Contains potentially unsupported claims")
        confidence -= 0.3
    
    confidence = max(confidence, 0.0)
    
    return {
        "has_issues": len(issues) > 0,
        "issues": issues,
        "confidence": confidence
    }


def format_sources(results: List[Dict]) -> List[Dict]:
    """
    Format source information for display.
    """
    sources = []
    for i, result in enumerate(results, 1):
        sources.append({
            "index": i,
            "content": result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
            "score": round(result.get('rerank_score', result['score']), 3),
            "metadata": {
                "source": result['source'],
                "page": result['page']
            }
        })
    return sources


def query_optimized_chain(chain, question: str, embed_model, index) -> Dict[str, Any]:
    """
    Execute optimized query with hybrid search, reranking, and validation.
    
    Args:
        chain: LangChain RAG chain
        question: User question
        embed_model: Embedding model
        index: Pinecone index
    
    Returns:
        Dictionary with answer, sources, and metadata
    """
    try:
        logger.info(f"Processing optimized query: {question}")
        
        # Step 1: Preprocess query
        from modules.optimized_llm import preprocess_query
        enhanced_query = preprocess_query(question)
        logger.info(f"Enhanced query: {enhanced_query}")
        
        # Step 2: Hybrid search
        retrieved_docs = hybrid_search(
            enhanced_query,
            embed_model,
            index,
            top_k=10
        )
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information in my knowledge base. Please try rephrasing your question or contact PEC directly for assistance.",
                "sources": [],
                "source_count": 0,
                "confidence": 0.0
            }
        
        # Step 3: Rerank results
        reranked_docs = rerank_results(enhanced_query, retrieved_docs, top_k=3)
        
        # Step 4: Prepare context for LLM
        context = "\n\n".join([
            f"[Source: {doc['source']}, Page: {doc['page']}]\n{doc['text']}"
            for doc in reranked_docs
        ])
        
        # Step 5: Generate answer using the chain
        # Create a simple retriever wrapper
        from langchain_core.documents import Document
        
        docs_for_chain = [
            Document(
                page_content=doc['text'],
                metadata={'source': doc['source'], 'page': doc['page']}
            )
            for doc in reranked_docs
        ]
        
        # Update the retriever's documents
        chain.steps[0]['context'].steps[0].documents = docs_for_chain
        
        # Invoke the chain
        answer = chain.invoke(enhanced_query)
        
        # Step 6: Validate and check answer quality
        from modules.optimized_llm import RAG_CONFIG
        
        quality_check = detect_answer_issues(answer, context, question)
        
        # Step 7: Enhance formatting
        from modules.optimized_llm import enhance_answer_with_formatting
        answer = enhance_answer_with_formatting(answer)
        
        # Step 8: Optional validation (if enabled)
        validation_result = None
        if RAG_CONFIG.get('enable_validation', False) and quality_check['confidence'] < 0.7:
            from modules.optimized_llm import validate_answer
            validation_result = validate_answer(question, context, answer)
            logger.info(f"Validation result: {validation_result}")
        
        # Step 9: Format response
        response = {
            "answer": answer,
            "sources": format_sources(reranked_docs),
            "source_count": len(reranked_docs),
            "confidence": quality_check['confidence'],
            "quality_issues": quality_check['issues'] if quality_check['has_issues'] else [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add validation info if performed
        if validation_result:
            response["validation"] = validation_result
        
        logger.info(f"Query completed with confidence: {quality_check['confidence']}")
        
        return response
        
    except Exception as e:
        logger.exception(f"Error in optimized query processing: {e}")
        return {
            "answer": f"An error occurred while processing your question. Please try again or rephrase your question.",
            "sources": [],
            "source_count": 0,
            "confidence": 0.0,
            "error": str(e)
        }
