#!/usr/bin/env python3
"""
Module to load scraped PEC data and prepare it for Pinecone upload.
Supports multiple sections: general, accreditation, cpd, engineers_registration, firm_registration
Compatible with: pinecone==5.4.2, langchain==0.2.14
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import time

logger = logging.getLogger(__name__)


def get_data_root() -> Path:
    """Get the absolute path to the data directory."""
    backend_dir = Path(__file__).parent.parent
    # Check both possible data locations
    data_root_option1 = backend_dir / "data" / "data"  # data/data/pec
    data_root_option2 = backend_dir / "data"  # data/pec
    
    # Check which one has the pec folder
    if (data_root_option1 / "pec").exists():
        return data_root_option1
    else:
        return data_root_option2


def get_all_section_folders() -> List[Path]:
    """Get all section folders in the PEC data directory."""
    data_root = get_data_root()
    pec_dir = data_root / "pec"
    
    # List of known section folders
    section_folders = []
    
    if pec_dir.exists():
        # Get all subdirectories
        for item in pec_dir.iterdir():
            if item.is_dir() and item.name != "__pycache__":
                section_folders.append(item)
    
    return section_folders


def get_scraped_data_stats() -> Dict[str, Any]:
    """Get statistics about scraped data available across all sections."""
    try:
        data_root = get_data_root()
        pec_dir = data_root / "pec"
        
        if not pec_dir.exists():
            return {
                "error": f"PEC directory not found: {pec_dir}",
                "total_chunks": 0
            }
        
        # Check for master index
        master_index_file = pec_dir / "all_sections_index.json"
        
        total_chunks = 0
        sections = {}
        section_details = {}
        
        # Get all section folders
        section_folders = get_all_section_folders()
        
        for section_folder in section_folders:
            section_name = section_folder.name
            
            # Check for two possible structures
            chunks_dir = section_folder / "chunks"
            
            if chunks_dir.exists():
                # Structure 1: Has chunks subfolder
                chunk_files = list(chunks_dir.glob("*.txt"))
                index_file = chunks_dir / "chunk_index.json"
                
                chunk_count = len(chunk_files)
                total_chunks += chunk_count
                sections[section_name] = chunk_count
                
                section_details[section_name] = {
                    "structure": "chunks subfolder",
                    "directory": str(chunks_dir.absolute()),
                    "chunk_count": chunk_count,
                    "index_exists": index_file.exists()
                }
            else:
                # Structure 2: Direct .txt files
                txt_files = list(section_folder.glob("*.txt"))
                
                chunk_count = len(txt_files)
                total_chunks += chunk_count
                sections[section_name] = chunk_count
                
                section_details[section_name] = {
                    "structure": "direct txt files",
                    "directory": str(section_folder.absolute()),
                    "chunk_count": chunk_count,
                    "index_exists": False
                }
        
        return {
            "total_chunks": total_chunks,
            "sections": sections,
            "section_details": section_details,
            "pec_directory": str(pec_dir.absolute()),
            "master_index_exists": master_index_file.exists()
        }
    
    except Exception as e:
        logger.error(f"Error getting scraped data stats: {e}")
        return {"error": str(e)}


def create_documents_from_chunks() -> List[Document]:
    """Create LangChain Document objects from all scraped chunks across all sections."""
    try:
        data_root = get_data_root()
        pec_dir = data_root / "pec"
        
        if not pec_dir.exists():
            logger.error(f"PEC directory not found: {pec_dir}")
            return []
        
        documents = []
        section_folders = get_all_section_folders()
        
        logger.info(f"Found {len(section_folders)} section folders to process")
        logger.info(f"PEC directory: {pec_dir}")
        
        for section_folder in section_folders:
            section_name = section_folder.name
            logger.info(f"Processing section: {section_name}")
            
            # Check for two possible structures:
            # 1. section/chunks/*.txt with chunk_index.json (like general)
            # 2. section/*.txt with direct files (like accreditation, cpd, etc.)
            
            chunks_dir = section_folder / "chunks"
            has_chunks_subfolder = chunks_dir.exists()
            
            if has_chunks_subfolder:
                # Structure 1: Has chunks subfolder
                index_file = chunks_dir / "chunk_index.json"
                
                if not index_file.exists():
                    logger.warning(f"Index file not found for section: {section_name}")
                    continue
                
                # Load chunk index
                try:
                    with open(index_file, 'r', encoding='utf-8') as f:
                        chunk_index = json.load(f)
                    
                    logger.info(f"Loading {len(chunk_index)} chunks from {section_name}/chunks/")
                    
                    for chunk_id, chunk_info in chunk_index.items():
                        chunk_file = chunks_dir / chunk_id
                        
                        try:
                            with open(chunk_file, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                            
                            if not content:
                                logger.warning(f"Empty content in chunk: {section_name}/{chunk_id}")
                                continue
                            
                            # Create metadata
                            metadata = {
                                'source': 'pec_scraped',
                                'section_folder': section_name,
                                'chunk_id': chunk_id,
                                'url': chunk_info.get('url', 'https://www.pec.org.pk'),
                                'title': chunk_info.get('title', 'PEC Website'),
                                'section': chunk_info.get('section', section_name),
                                'timestamp': chunk_info.get('fetched_at', ''),
                                'origin_file': chunk_info.get('origin_file', ''),
                            }
                            
                            doc = Document(page_content=content, metadata=metadata)
                            documents.append(doc)
                            
                        except Exception as e:
                            logger.error(f"Error reading chunk {section_name}/{chunk_file.name}: {e}")
                            continue
                
                except Exception as e:
                    logger.error(f"Error processing section {section_name}: {e}")
                    continue
            
            else:
                # Structure 2: Direct .txt files in section folder
                txt_files = list(section_folder.glob("*.txt"))
                
                if not txt_files:
                    logger.warning(f"No .txt files found in section: {section_name}")
                    continue
                
                logger.info(f"Loading {len(txt_files)} direct .txt files from {section_name}/")
                
                for txt_file in txt_files:
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        
                        if not content:
                            logger.warning(f"Empty content in file: {section_name}/{txt_file.name}")
                            continue
                        
                        # Create metadata for direct files
                        metadata = {
                            'source': 'pec_scraped',
                            'section_folder': section_name,
                            'chunk_id': txt_file.name,
                            'url': f'https://www.pec.org.pk/{section_name}',
                            'title': f'PEC {section_name.replace("_", " ").title()}',
                            'section': section_name,
                            'timestamp': '',
                            'origin_file': txt_file.name,
                        }
                        
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
                        
                    except Exception as e:
                        logger.error(f"Error reading file {section_name}/{txt_file.name}: {e}")
                        continue
        
        logger.info(f"Successfully created {len(documents)} documents from all sections")
        return documents
    
    except Exception as e:
        logger.error(f"Error creating documents from chunks: {e}")
        logger.exception("Full traceback:")
        return []


def upload_scraped_data_to_pinecone(
    pinecone_api_key: str,
    pinecone_index_name: str = "pec-assistant-index",
    batch_size: int = 100
) -> bool:
    """
    Upload scraped data to Pinecone vector database.
    Uses HuggingFace embeddings (sentence-transformers) since OpenAI is not installed.
    
    Args:
        pinecone_api_key: Pinecone API key
        pinecone_index_name: Name of the Pinecone index
        batch_size: Number of documents to upload per batch
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Starting Pinecone upload process")
        
        # Create documents from chunks
        documents = create_documents_from_chunks()
        
        if not documents:
            logger.error("No documents to upload")
            return False
        
        logger.info(f"Preparing to upload {len(documents)} documents")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize embeddings using HuggingFace (sentence-transformers)
        logger.info("Loading embedding model (this may take a moment)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Get embedding dimension
        test_embedding = embeddings.embed_query("test")
        embedding_dimension = len(test_embedding)
        logger.info(f"Embedding dimension: {embedding_dimension}")
        
        # Check if index exists, create if not
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if pinecone_index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {pinecone_index_name}")
            pc.create_index(
                name=pinecone_index_name,
                dimension=embedding_dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            logger.info("Waiting for index to be ready...")
            time.sleep(10)
        else:
            logger.info(f"Using existing index: {pinecone_index_name}")
        
        # Get index
        index = pc.Index(pinecone_index_name)
        
        # Upload in batches
        logger.info("Creating embeddings and uploading to Pinecone...")
        total_uploaded = 0
        section_counts = {}
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            # Prepare vectors for upload
            vectors_to_upsert = []
            
            for doc_idx, doc in enumerate(batch):
                try:
                    # Create embedding
                    embedding = embeddings.embed_query(doc.page_content)
                    
                    # Create unique ID with section prefix
                    section_folder = doc.metadata.get('section_folder', 'unknown')
                    chunk_id = doc.metadata.get('chunk_id', f'doc_{total_uploaded + doc_idx}')
                    vector_id = f"{section_folder}--{chunk_id}"
                    
                    # Track section counts
                    section_counts[section_folder] = section_counts.get(section_folder, 0) + 1
                    
                    # Prepare metadata (Pinecone has limitations on metadata)
                    vector_metadata = {
                        'text': doc.page_content[:1000],  # Limit text length
                        'source': doc.metadata.get('source', ''),
                        'section_folder': section_folder,
                        'url': doc.metadata.get('url', ''),
                        'title': doc.metadata.get('title', ''),
                        'section': doc.metadata.get('section', ''),
                        'origin_file': doc.metadata.get('origin_file', ''),
                    }
                    
                    vectors_to_upsert.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': vector_metadata
                    })
                    
                except Exception as e:
                    logger.error(f"Error creating embedding for document {doc_idx}: {e}")
                    continue
            
            # Upsert to Pinecone
            if vectors_to_upsert:
                try:
                    index.upsert(vectors=vectors_to_upsert)
                    total_uploaded += len(vectors_to_upsert)
                    logger.info(f"Progress: {total_uploaded}/{len(documents)} documents uploaded")
                except Exception as e:
                    logger.error(f"Error upserting batch {batch_num}: {e}")
                    continue
            
            # Small delay between batches to avoid rate limits
            if i + batch_size < len(documents):
                time.sleep(0.5)
        
        logger.info(f"✅ Successfully uploaded {total_uploaded} documents to Pinecone")
        logger.info(f"Documents per section: {section_counts}")
        
        # Verify upload
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error uploading to Pinecone: {e}")
        logger.exception("Full traceback:")
        return False


def upload_filtered_documents_to_pinecone(
    documents: List[Document],
    pinecone_api_key: str,
    pinecone_index_name: str = "pec-assistant-index",
    batch_size: int = 100
) -> bool:
    """
    Upload a filtered list of documents to Pinecone.
    Useful for uploading only specific sections.
    
    Args:
        documents: List of Document objects to upload
        pinecone_api_key: Pinecone API key
        pinecone_index_name: Name of the Pinecone index
        batch_size: Number of documents to upload per batch
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Starting upload of {len(documents)} filtered documents")
        
        if not documents:
            logger.error("No documents provided")
            return False
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize embeddings
        logger.info("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Get embedding dimension
        test_embedding = embeddings.embed_query("test")
        embedding_dimension = len(test_embedding)
        logger.info(f"Embedding dimension: {embedding_dimension}")
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if pinecone_index_name not in existing_indexes:
            logger.error(f"Index '{pinecone_index_name}' does not exist!")
            return False
        
        logger.info(f"Using existing index: {pinecone_index_name}")
        
        # Get index
        index = pc.Index(pinecone_index_name)
        
        # Upload in batches
        logger.info("Creating embeddings and uploading to Pinecone...")
        total_uploaded = 0
        section_counts = {}
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            # Prepare vectors for upload
            vectors_to_upsert = []
            
            for doc_idx, doc in enumerate(batch):
                try:
                    # Create embedding
                    embedding = embeddings.embed_query(doc.page_content)
                    
                    # Create unique ID with section prefix
                    section_folder = doc.metadata.get('section_folder', 'unknown')
                    chunk_id = doc.metadata.get('chunk_id', f'doc_{total_uploaded + doc_idx}')
                    vector_id = f"{section_folder}--{chunk_id}"
                    
                    # Track section counts
                    section_counts[section_folder] = section_counts.get(section_folder, 0) + 1
                    
                    # Prepare metadata
                    vector_metadata = {
                        'text': doc.page_content[:1000],
                        'source': doc.metadata.get('source', ''),
                        'section_folder': section_folder,
                        'url': doc.metadata.get('url', ''),
                        'title': doc.metadata.get('title', ''),
                        'section': doc.metadata.get('section', ''),
                        'origin_file': doc.metadata.get('origin_file', ''),
                    }
                    
                    vectors_to_upsert.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': vector_metadata
                    })
                    
                except Exception as e:
                    logger.error(f"Error creating embedding for document {doc_idx}: {e}")
                    continue
            
            # Upsert to Pinecone
            if vectors_to_upsert:
                try:
                    index.upsert(vectors=vectors_to_upsert)
                    total_uploaded += len(vectors_to_upsert)
                    logger.info(f"Progress: {total_uploaded}/{len(documents)} documents uploaded")
                except Exception as e:
                    logger.error(f"Error upserting batch {batch_num}: {e}")
                    continue
            
            # Small delay between batches
            if i + batch_size < len(documents):
                time.sleep(0.5)
        
        logger.info(f"✅ Successfully uploaded {total_uploaded} documents")
        logger.info(f"Documents per section: {section_counts}")
        
        # Verify upload
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error uploading filtered documents: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    # Test the module
    print("Testing load_scraped_data module...")
    print("\nStatistics:")
    stats = get_scraped_data_stats()
    for key, value in stats.items():
        if key == "section_details":
            print(f"  {key}:")
            for section, details in value.items():
                print(f"    {section}:")
                for detail_key, detail_value in details.items():
                    print(f"      - {detail_key}: {detail_value}")
        else:
            print(f"  {key}: {value}")
    
    print("\nCreating documents...")
    docs = create_documents_from_chunks()
    print(f"Created {len(docs)} documents")
    
    if docs:
        print("\nSample document:")
        print(f"  Content length: {len(docs[0].page_content)}")
        print(f"  Metadata: {docs[0].metadata}")
        print(f"  Content preview: {docs[0].page_content[:200]}...")