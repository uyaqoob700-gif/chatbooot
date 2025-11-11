from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from typing import Dict, Any, List
import logging
import re
from pinecone import Pinecone
import os

logger = logging.getLogger(__name__)


def expand_query_for_retrieval(question: str) -> List[str]:
    """Expand query with related terms for better retrieval."""
    
    # PEC-specific query expansion
    expansion_map = {
        'registration': ['registration', 'register', 'enrollment', 'enroll', 'apply', 'application'],
        'engineer': ['engineer', 'engineering', 'professional engineer', 'pe', 'registered engineer', 're'],
        'firm': ['firm', 'company', 'consultant', 'contractor', 'organization', 'enterprise'],
        'examination': ['examination', 'exam', 'test', 'epe', 'engineering practice examination', 'assessment'],
        'cpd': ['cpd', 'continuing professional development', 'training', 'course', 'education', 'learning'],
        'fee': ['fee', 'fees', 'payment', 'cost', 'charge', 'amount'],
        'renewal': ['renewal', 'renew', 'expiry', 'expire', 'validity', 'valid'],
        'regulation': ['regulation', 'regulations', 'act', 'rule', 'rules', 'policy', 'policies'],
        'requirement': ['requirement', 'requirements', 'criteria', 'condition', 'prerequisite'],
        'document': ['document', 'documents', 'paper', 'certificate', 'license', 'permit'],
        'procedure': ['procedure', 'procedures', 'process', 'steps', 'method', 'way']
    }
    
    expanded_queries = [question]
    question_lower = question.lower()
    
    # Add related terms
    for key, terms in expansion_map.items():
        if key in question_lower:
            for term in terms:
                if term not in question_lower:
                    expanded_queries.append(f"{question} {term}")
    
    # Add year-based queries if year is mentioned
    year_match = re.search(r'\b(20\d{2})\b', question)
    if year_match:
        year = year_match.group(1)
        expanded_queries.append(f"{question} {year}")
    
    return expanded_queries[:3]  # Limit to 3 expanded queries


def hybrid_search_pinecone(question: str, embed_model, index, top_k: int = 10) -> List[Document]:
    """Perform hybrid search combining multiple strategies."""
    
    # Expand query
    expanded_queries = expand_query_for_retrieval(question)
    
    all_results = []
    seen_ids = set()
    
    for query in expanded_queries:
        try:
            # Generate embedding for this query
            query_embedding = embed_model.embed_query(query)
            
            # Search Pinecone
            search_results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={}  # No filter for now, can add document_type filter later
            )
            
            # Process results
            for match in search_results.matches:
                if match.id not in seen_ids:
                    doc = Document(
                        page_content=match.metadata.get('text', ''),
                        metadata={
                            'source_file': match.metadata.get('source_file', 'unknown'),
                            'score': match.score,
                            'chunk_index': match.metadata.get('chunk_index', 0),
                            'document_type': match.metadata.get('document_type', 'general'),
                            'key_terms': match.metadata.get('key_terms', []),
                            'query_used': query
                        }
                    )
                    all_results.append(doc)
                    seen_ids.add(match.id)
                    
        except Exception as e:
            logger.warning(f"Error in hybrid search for query '{query}': {e}")
            continue
    
    # Sort by score and return top results
    all_results.sort(key=lambda x: x.metadata['score'], reverse=True)
    return all_results[:top_k]


def rerank_documents(documents: List[Document], question: str) -> List[Document]:
    """Rerank documents based on relevance to the question."""
    
    question_lower = question.lower()
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    
    scored_docs = []
    
    for doc in documents:
        content_lower = doc.page_content.lower()
        content_words = set(re.findall(r'\b\w+\b', content_lower))
        
        # Calculate relevance score
        word_overlap = len(question_words.intersection(content_words))
        word_overlap_ratio = word_overlap / len(question_words) if question_words else 0
        
        # Boost score for PEC-specific terms
        pec_terms = ['pec', 'pakistan engineering council', 'engineering', 'professional', 'registration']
        pec_boost = sum(1 for term in pec_terms if term in content_lower)
        
        # Boost score for document type relevance
        doc_type_boost = 0
        if 'registration' in question_lower and doc.metadata.get('document_type') == 'procedure':
            doc_type_boost = 0.2
        elif 'examination' in question_lower and doc.metadata.get('document_type') == 'examination':
            doc_type_boost = 0.2
        elif 'cpd' in question_lower and doc.metadata.get('document_type') == 'cpd':
            doc_type_boost = 0.2
        
        # Combined score
        relevance_score = (
            doc.metadata.get('score', 0) * 0.4 +  # Original similarity score
            word_overlap_ratio * 0.3 +            # Word overlap
            pec_boost * 0.1 +                     # PEC term boost
            doc_type_boost                         # Document type boost
        )
        
        doc.metadata['relevance_score'] = relevance_score
        scored_docs.append(doc)
    
    # Sort by relevance score
    scored_docs.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
    return scored_docs


def query_optimized_chain(chain: RetrievalQA, question: str, embed_model, index) -> Dict[str, Any]:
    """
    Query the optimized RAG chain with hybrid search and reranking.
    """
    try:
        logger.info(f"Querying optimized chain with question: {question}")
        
        # Perform hybrid search
        retrieved_docs = hybrid_search_pinecone(question, embed_model, index, top_k=15)
        
        if not retrieved_docs:
            logger.warning("No relevant documents found in hybrid search")
            return {
                "answer": "I couldn't find any relevant information in the uploaded PEC documents to answer your question. Please make sure documents have been uploaded and try rephrasing your question.",
                "sources": [],
                "source_count": 0,
                "search_method": "hybrid_search"
            }
        
        # Rerank documents
        reranked_docs = rerank_documents(retrieved_docs, question)
        
        # Use top 5 documents for context (reduced from 8 for more focused responses)
        top_docs = reranked_docs[:5]
        
        # Create a custom retriever with the top documents
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from typing import List
        
        class SimpleRetriever(BaseRetriever):
            """Custom retriever that returns pre-fetched documents."""
            documents: List[Document] = []
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                """Return the stored documents."""
                return self.documents
            
            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                """Async version of get_relevant_documents."""
                return self.documents
        custom_retriever = SimpleRetriever(documents=top_docs)
        
        # Update the chain's retriever temporarily
        original_retriever = chain.retriever
        chain.retriever = custom_retriever
        
        try:
            # Query the chain
            response = chain({"query": question})
            
            # Extract answer
            answer = response.get("result", "")
            
            # Extract and format sources
            sources = []
            source_docs = response.get("source_documents", [])
            
            for i, doc in enumerate(source_docs[:5]):  # Show top 5 sources
                source_info = {
                    "index": i + 1,
                    "content": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content,
                    "metadata": {
                        'source_file': doc.metadata.get('source_file', 'unknown'),
                        'relevance_score': doc.metadata.get('relevance_score', 0),
                        'document_type': doc.metadata.get('document_type', 'general'),
                        'chunk_index': doc.metadata.get('chunk_index', 0)
                    }
                }
                sources.append(source_info)
            
            logger.info(f"Successfully generated optimized answer with {len(source_docs)} sources")
            
            return {
                "answer": answer,
                "sources": sources,
                "source_count": len(source_docs),
                "search_method": "hybrid_search_with_reranking",
                "total_documents_searched": len(retrieved_docs)
            }
            
        finally:
            # Restore original retriever
            chain.retriever = original_retriever
        
    except Exception as e:
        logger.error(f"Error in optimized query: {e}")
        raise


def query_standard_chain(chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """
    Standard query handler (fallback).
    """
    try:
        logger.info(f"Querying standard chain with question: {question}")
        
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
        
        logger.info(f"Successfully generated standard answer with {len(source_docs)} sources")
        
        return {
            "answer": answer,
            "sources": sources,
            "source_count": len(source_docs),
            "search_method": "standard_search"
        }
        
    except Exception as e:
        logger.error(f"Error querying standard chain: {e}")
        raise
