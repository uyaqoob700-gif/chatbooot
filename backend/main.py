from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.load_vectorstore import load_vectorstore
from typing import List, Optional
from pydantic import Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from modules.intent_router import classify_query_with_groq

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Configuration
USE_MOCK = os.getenv("USE_MOCK", "False").lower() == "true"  # Default to production

# Conversation logging directory
CONVERSATION_LOG_DIR = Path(__file__).parent / "conversation_logs"
CONVERSATION_LOG_DIR.mkdir(exist_ok=True)

def log_conversation(question: str, answer: str, sources: list = None, metadata: dict = None):
    """
    Log conversation data for future training and analysis.
    Saves to a JSON file with timestamp.
    """
    try:
        timestamp = datetime.now()
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "metadata": metadata or {}
        }
        
        # Create daily log file
        log_filename = f"conversations_{timestamp.strftime('%Y-%m-%d')}.json"
        log_filepath = CONVERSATION_LOG_DIR / log_filename
        
        # Load existing conversations or create new list
        if log_filepath.exists():
            with open(log_filepath, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        else:
            conversations = []
        
        # Append new conversation
        conversations.append(log_entry)
        
        # Save back to file
        with open(log_filepath, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversation logged to {log_filename}")
        
    except Exception as e:
        logger.error(f"Error logging conversation: {e}")

# Validate environment variables for production mode
if not USE_MOCK:
    required_vars = ["GROQ_API_KEY", "PINECONE_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.warning("Falling back to MOCK mode")
        USE_MOCK = True

app = FastAPI(
    title="RagBot2.0",
    description="RAG-based chatbot with PDF upload and query capabilities",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Custom Retriever Class
class SimpleRetriever(BaseRetriever):
    """Custom retriever that returns pre-fetched documents."""
    
    documents: List[Document] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Return the stored documents."""
        return self.documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self.documents


@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
    """Global exception handler middleware."""
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred"
            }
        )


@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload and process PDF/TXT files to add to the vector store.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate file types (accepts both PDF and TXT)
        for file in files:
            if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.filename}. Only PDF and TXT files are allowed."
                )
        
        logger.info(f"Received {len(files)} files for upload")
        
        # In MOCK mode, just simulate processing
        if USE_MOCK:
            logger.info("[MOCK MODE] Simulating file processing...")
            for file in files:
                content = await file.read()
                logger.info(f"  - Received: {file.filename} ({len(content)} bytes)")
            logger.info("[MOCK MODE] In production, documents would be:")
            logger.info("  1. Extracted and split into chunks")
            logger.info("  2. Converted to embeddings (using HuggingFace)")
            logger.info("  3. Stored in Pinecone vector database")
            
            return {
                "message": "Files processed successfully [MOCK MODE]",
                "files_processed": len(files),
                "filenames": [f.filename for f in files],
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Real processing - Upload to Pinecone with HuggingFace embeddings
            logger.info("[PRODUCTION MODE] Processing files and uploading to Pinecone...")
            
            try:
                # Use existing vectorstore loader to upsert into Pinecone
                load_vectorstore(files)
                
                logger.info(f"✅ Successfully uploaded {len(files)} files to Pinecone")
                
                return {
                    "message": "Files processed and uploaded to Pinecone successfully",
                    "files_processed": len(files),
                    "filenames": [f.filename for f in files],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error uploading to Pinecone: {str(e)}")
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error uploading to vector store: {str(e)}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during file upload")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    """
    Ask a question and get an answer based on the uploaded documents.
    Uses Pinecone for retrieval and Groq for answer generation.
    """
    try:
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"User query: {question}")

        # Intent routing: handle capability queries separately; allow greetings to go to LLM
        normalized = question.strip().lower()
        capability_intents = [
            "how can you help",
            "what can you do",
            "how do you help",
            "who are you",
            "what is your purpose",
            "help",
            "what can u do",
            "how can u help"
        ]
        if any(phrase in normalized for phrase in capability_intents):
            capability_text = (
                "I can help you with Engineer Registration, Firm Licensing, CPD, "
                "Project Approvals, and other PEC services. Ask me about eligibility, "
                "required documents, fees, timelines, or how to apply."
            )
            response = {
                "answer": capability_text,
                "sources": [],
                "source_count": 0,
                "mode": "capability",
                "timestamp": datetime.now().isoformat()
            }
            # Log and return immediately
            log_conversation(
                question=question,
                answer=response["answer"],
                sources=response["sources"],
                metadata={"mode": "capability", "source_count": 0, "timestamp": response["timestamp"]}
            )
            return response

        # Simple small-talk handling: defer to LLM in production, static in mock
        greeting_patterns = {"hi", "hello", "hey", "good morning", "good evening"}
        if normalized in greeting_patterns or any(normalized.startswith(g + " ") for g in greeting_patterns):
            if USE_MOCK:
                smalltalk = "Hello! How can I assist you with PEC services today?"
                log_conversation(question=question, answer=smalltalk, sources=[], metadata={"mode": "smalltalk-mock", "source_count": 0, "timestamp": datetime.now().isoformat()})
                return {"answer": smalltalk, "sources": [], "source_count": 0, "mode": "smalltalk"}
            else:
                try:
                    from langchain_groq import ChatGroq
                    from langchain_core.messages import SystemMessage, HumanMessage
                    llm = ChatGroq(
                        api_key=os.environ.get("GROQ_API_KEY"),
                        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                        temperature=0.7,
                        max_tokens=256
                    )
                    messages = [
                        SystemMessage(content="You are a friendly PEC assistant. Keep replies brief and helpful."),
                        HumanMessage(content=question)
                    ]
                    reply = llm.invoke(messages).content
                    result = {"answer": reply, "sources": [], "source_count": 0, "mode": "smalltalk", "timestamp": datetime.now().isoformat()}
                    log_conversation(question=question, answer=reply, sources=[], metadata={"mode": "smalltalk", "source_count": 0, "timestamp": result["timestamp"]})
                    return result
                except Exception as e:
                    logger.warning(f"Smalltalk LLM failed, falling back to static: {e}")
                    fallback = "Hello! How can I assist you with PEC services today?"
                    return {"answer": fallback, "sources": [], "source_count": 0, "mode": "smalltalk"}

        # If not capability or greeting, use LLM intent classifier to route
        intent = classify_query_with_groq(question) if not USE_MOCK else "rag"
        logger.info(f"Intent classified as: {intent}")
        if intent == "general" and not USE_MOCK:
            try:
                from langchain_groq import ChatGroq
                from langchain_core.messages import SystemMessage, HumanMessage
                llm = ChatGroq(
                    api_key=os.environ.get("GROQ_API_KEY"),
                    model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    temperature=0.3,
                    max_tokens=512
                )
                messages = [
                    SystemMessage(content=(
                        "You are a helpful PEC assistant. Answer accurately and concisely. "
                        "If the user asks for specific official rules/fees/documents not provided, suggest checking uploaded documents."
                    )),
                    HumanMessage(content=question)
                ]
                reply = llm.invoke(messages).content
                result = {"answer": reply, "sources": [], "source_count": 0, "mode": "general", "timestamp": datetime.now().isoformat()}
                log_conversation(question=question, answer=reply, sources=[], metadata={"mode": "general", "source_count": 0, "timestamp": result["timestamp"]})
                return result
            except Exception as e:
                logger.warning(f"General LLM answer failed, will fall back to RAG: {e}")
                # Continue into RAG flow below

        # Mock mode
        if USE_MOCK:
            mock_response = {
                "answer": f"[MOCK MODE] You asked: '{question}'. In production with Groq API, the bot would search through uploaded documents in Pinecone and provide an intelligent answer based on the content.",
                "sources": [
                    {
                        "index": 1,
                        "content": "Sample content from document 1...",
                        "metadata": {"source": "example.pdf", "page": 1}
                    }
                ],
                "source_count": 1,
                "mode": "mock",
                "timestamp": datetime.now().isoformat()
            }
            logger.info("[MOCK MODE] Returning mock response")
            
            # Log conversation even in mock mode
            log_conversation(
                question=question,
                answer=mock_response["answer"],
                sources=mock_response["sources"],
                metadata={
                    "source_count": mock_response["source_count"],
                    "mode": "mock",
                    "timestamp": mock_response["timestamp"]
                }
            )
            
            return mock_response

        # PRODUCTION MODE with Optimized Pinecone + Groq
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from pinecone import Pinecone
            from modules.optimized_llm import get_optimized_llm_chain
            from modules.optimized_query_handlers import query_optimized_chain
            
            logger.info("[OPTIMIZED PRODUCTION MODE] Using enhanced RAG system")
            
            # Initialize embeddings (same as used for upload)
            embed_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "pec-assistant-index"))
            
            # Create optimized chain (will be used internally by query handler)
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
            
            dummy_retriever = SimpleRetriever(documents=[])
            chain = get_optimized_llm_chain(dummy_retriever, temperature=0.3)
            
            # Use optimized query handler with hybrid search and reranking
            result = query_optimized_chain(chain, question, embed_model, index)
            
            logger.info("Optimized query successful - Generated enhanced AI response")
            
            # Add timestamp to result
            result["timestamp"] = datetime.now().isoformat()
            
            # Log conversation for future training
            log_conversation(
                question=question,
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                metadata={
                    "source_count": result.get("source_count", 0),
                    "timestamp": result.get("timestamp", ""),
                    "mode": "production"
                }
            )
            
            return result
            
        except ImportError as e:
            logger.error(f"Missing dependencies for production mode: {e}")
            raise HTTPException(
                status_code=500,
                detail="Production mode requires additional packages. Install: pip install langchain-groq langchain-huggingface sentence-transformers pinecone-client"
            )
        except Exception as e:
            logger.exception("Error in production mode")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing with Groq API: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing question")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RagBot2.0",
        "version": "2.0.0",
        "mode": "mock" if USE_MOCK else "production",
        "status": "operational",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "pinecone_configured": bool(os.getenv("PINECONE_API_KEY")),
        "embeddings": "HuggingFace (sentence-transformers/all-MiniLM-L6-v2)",
        "endpoints": {
            "/": "GET - API information",
            "/health": "GET - Health check",
            "/upload_pdfs/": "POST - Upload PDF/TXT files",
            "/ask/": "POST - Ask a question",
            "/test": "GET - Test endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mode": "mock" if USE_MOCK else "production",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "pinecone_configured": bool(os.getenv("PINECONE_API_KEY")),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/test")
async def test():
    """Simple test endpoint."""
    return {
        "message": "Testing successful!",
        "mode": "mock" if USE_MOCK else "production",
        "embeddings": "HuggingFace (local, no API limits)"
    }


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 80)
    logger.info("RagBot2.0 Backend Starting...")
    logger.info(f"Mode: {'MOCK' if USE_MOCK else 'PRODUCTION'}")
    logger.info(f"Groq API Key: {'✓ Configured' if os.getenv('GROQ_API_KEY') else '✗ Missing'}")
    logger.info(f"Pinecone API Key: {'✓ Configured' if os.getenv('PINECONE_API_KEY') else '✗ Missing'}")
    logger.info(f"Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2) - OPTIMIZED")
    logger.info(f"Vector Store: Pinecone ({os.getenv('PINECONE_INDEX_NAME', 'pec-assistant-index')}) - ENHANCED")
    logger.info(f"RAG System: Hybrid Search + Reranking + PEC-Specialized Prompts")
    logger.info("=" * 80)
    
    if USE_MOCK:
        logger.warning("⚠️  Running in MOCK MODE")
    else:
        logger.info("✓ Running in PRODUCTION MODE with Groq API + Pinecone + HuggingFace")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)