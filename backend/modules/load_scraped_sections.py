#!/usr/bin/env python3
"""
Module to load data from data/scraped_data/pec/ folders
This handles the section folders: accreditation, cpd, engineers_registration, firm_registration, general
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


def get_scraped_data_root() -> Path:
    """Get the absolute path to the scraped_data directory."""
    backend_dir = Path(__file__).parent.parent
    project_dir = backend_dir.parent
    
    # The scraped data is in: project/data/scraped_data
    scraped_data_path = project_dir / "data" / "scraped_data"
    
    if not scraped_data_path.exists():
        logger.warning(f"Scraped data path not found: {scraped_data_path}")
    
    return scraped_data_path


def get_section_folders() -> List[Path]:
    """Get all section folders from scraped_data/pec/"""
    scraped_root = get_scraped_data_root()
    pec_dir = scraped_root / "pec"
    
    section_folders = []
    
    if pec_dir.exists():
        for item in pec_dir.iterdir():
            if item.is_dir() and item.name != "__pycache__":
                section_folders.append(item)
    
    return section_folders


def get_scraped_sections_stats() -> Dict[str, Any]:
    """Get statistics about scraped section data."""
    try:
        scraped_root = get_scraped_data_root()
        pec_dir = scraped_root / "pec"
        
        if not pec_dir.exists():
            return {
                "error": f"Scraped PEC directory not found: {pec_dir}",
                "total_files": 0
            }
        
        # Check for master index
        master_index_file = pec_dir / "all_sections_index.json"
        
        total_files = 0
        sections = {}
        section_details = {}
        
        # Get all section folders
        section_folders = get_section_folders()
        
        for section_folder in section_folders:
            section_name = section_folder.name
            
            # Count .txt files in the section folder
            txt_files = list(section_folder.glob("*.txt"))
            
            file_count = len(txt_files)
            total_files += file_count
            sections[section_name] = file_count
            
            section_details[section_name] = {
                "directory": str(section_folder.absolute()),
                "file_count": file_count,
                "sample_files": [f.name for f in txt_files[:3]]
            }
        
        return {
            "total_files": total_files,
            "sections": sections,
            "section_details": section_details,
            "pec_directory": str(pec_dir.absolute()),
            "master_index_exists": master_index_file.exists()
        }
    
    except Exception as e:
        logger.error(f"Error getting scraped sections stats: {e}")
        return {"error": str(e)}


def create_documents_from_scraped_sections(exclude_sections: List[str] = None) -> List[Document]:
    """
    Create LangChain Document objects from scraped section .txt files.
    
    Args:
        exclude_sections: List of section names to skip (e.g., ['general'] if already uploaded)
    
    Returns:
        List of Document objects
    """
    try:
        if exclude_sections is None:
            exclude_sections = []
        
        scraped_root = get_scraped_data_root()
        pec_dir = scraped_root / "pec"
        
        if not pec_dir.exists():
            logger.error(f"Scraped PEC directory not found: {pec_dir}")
            return []
        
        documents = []
        section_folders = get_section_folders()
        
        logger.info(f"Found {len(section_folders)} section folders in scraped_data")
        logger.info(f"Scraped PEC directory: {pec_dir}")
        
        for section_folder in section_folders:
            section_name = section_folder.name
            
            # Skip excluded sections
            if section_name in exclude_sections:
                logger.info(f"Skipping section: {section_name} (excluded)")
                continue
            
            logger.info(f"Processing section: {section_name}")
            
            # Get all .txt files in the section folder
            txt_files = list(section_folder.glob("*.txt"))
            
            if not txt_files:
                logger.warning(f"No .txt files found in section: {section_name}")
                continue
            
            logger.info(f"Loading {len(txt_files)} files from {section_name}/")
            
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if not content:
                        logger.warning(f"Empty content in file: {section_name}/{txt_file.name}")
                        continue
                    
                    # Create metadata
                    metadata = {
                        'source': 'pec_scraped_sections',
                        'section_folder': section_name,
                        'file_name': txt_file.name,
                        'url': f'https://www.pec.org.pk/{section_name}',
                        'title': f'PEC {section_name.replace("_", " ").title()} - {txt_file.stem}',
                        'section': section_name,
                    }
                    
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error reading file {section_name}/{txt_file.name}: {e}")
                    continue
        
        logger.info(f"Successfully created {len(documents)} documents from scraped sections")
        return documents
    
    except Exception as e:
        logger.error(f"Error creating documents from scraped sections: {e}")
        logger.exception("Full traceback:")
        return []


def upload_scraped_sections_to_pinecone(
    pinecone_api_key: str,
    pinecone_index_name: str = "pec-assistant-index",
    exclude_sections: List[str] = None,
    batch_size: int = 50
) -> bool:
    """
    Upload scraped section data to Pinecone.
    
    Args:
        pinecone_api_key: Pinecone API key
        pinecone_index_name: Name of the Pinecone index
        exclude_sections: List of sections to skip
        batch_size: Number of documents per batch
    
    Returns:
        bool: True if successful
    """
    try:
        logger.info("Starting Pinecone upload for scraped sections")
        
        # Create documents
        documents = create_documents_from_scraped_sections(exclude_sections=exclude_sections)
        
        if not documents:
            logger.error("No documents to upload")
            return False
        
        logger.info(f"Preparing to upload {len(documents)} documents")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize embeddings
        logger.info("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if pinecone_index_name not in existing_indexes:
            logger.error(f"Index '{pinecone_index_name}' does not exist!")
            return False
        
        # Get index
        index = pc.Index(pinecone_index_name)
        
        # Upload in batches
        logger.info("Creating embeddings and uploading...")
        total_uploaded = 0
        section_counts = {}
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            vectors_to_upsert = []
            
            for doc_idx, doc in enumerate(batch):
                try:
                    # Create embedding
                    embedding = embeddings.embed_query(doc.page_content)
                    
                    # Create unique ID
                    section_folder = doc.metadata.get('section_folder', 'unknown')
                    file_name = doc.metadata.get('file_name', f'doc_{total_uploaded + doc_idx}')
                    vector_id = f"scraped-{section_folder}--{file_name}"
                    
                    # Track section counts
                    section_counts[section_folder] = section_counts.get(section_folder, 0) + 1
                    
                    # Prepare metadata
                    vector_metadata = {
                        'text': doc.page_content[:1000],
                        'source': doc.metadata.get('source', ''),
                        'section_folder': section_folder,
                        'file_name': file_name,
                        'url': doc.metadata.get('url', ''),
                        'title': doc.metadata.get('title', ''),
                        'section': doc.metadata.get('section', ''),
                    }
                    
                    vectors_to_upsert.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': vector_metadata
                    })
                    
                except Exception as e:
                    logger.error(f"Error creating embedding: {e}")
                    continue
            
            # Upsert batch
            if vectors_to_upsert:
                try:
                    index.upsert(vectors=vectors_to_upsert)
                    total_uploaded += len(vectors_to_upsert)
                    logger.info(f"Progress: {total_uploaded}/{len(documents)} uploaded")
                except Exception as e:
                    logger.error(f"Error upserting batch {batch_num}: {e}")
                    continue
            
            if i + batch_size < len(documents):
                time.sleep(0.5)
        
        logger.info(f"✅ Successfully uploaded {total_uploaded} documents")
        logger.info(f"Documents per section: {section_counts}")
        
        # Verify
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error uploading scraped sections: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    # Test the module
    print("Testing scraped sections loader...")
    print("\nStatistics:")
    stats = get_scraped_sections_stats()
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
    docs = create_documents_from_scraped_sections()
    print(f"Created {len(docs)} documents")
    
    if docs:
        print("\nSample document:")
        print(f"  Section: {docs[0].metadata.get('section_folder')}")
        print(f"  File: {docs[0].metadata.get('file_name')}")
        print(f"  Content length: {len(docs[0].page_content)}")
        print(f"  Content preview: {docs[0].page_content[:200]}...")