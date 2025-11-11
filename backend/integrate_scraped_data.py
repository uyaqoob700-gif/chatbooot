#!/usr/bin/env python3
"""
Integration script to load scraped PEC data into Pinecone vector database.
This script integrates all the chunks and indexes from the data folder into the backend.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the backend modules to path
sys.path.append(str(Path(__file__).parent))

from modules.load_scraped_data import (
    upload_scraped_data_to_pinecone,
    get_scraped_data_stats,
    create_documents_from_chunks
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main integration function."""
    print("=" * 80)
    print("PEC SCRAPED DATA INTEGRATION TO PINECONE")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "pec-assistant-index")
    
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY not found in environment variables")
        logger.error("Please set PINECONE_API_KEY in your .env file")
        return False
    
    print(f"Pinecone Index: {pinecone_index_name}")
    print(f"API Key: {'*' * (len(pinecone_api_key) - 4) + pinecone_api_key[-4:]}")
    print()
    
    # Get scraped data statistics
    print("📊 SCRAPED DATA STATISTICS")
    print("-" * 40)
    stats = get_scraped_data_stats()
    
    if "error" in stats:
        logger.error(f"Error getting stats: {stats['error']}")
        return False
    
    for key, value in stats.items():
        if key == "sections":
            print(f"  {key}:")
            for section, count in value.items():
                print(f"    - {section}: {count} chunks")
        else:
            print(f"  {key}: {value}")
    
    print()
    
    # Create documents preview
    print("📄 DOCUMENT PREVIEW")
    print("-" * 40)
    documents = create_documents_from_chunks()
    
    if not documents:
        logger.error("No documents found to upload")
        return False
    
    print(f"Total documents created: {len(documents)}")
    
    # Show sample documents
    for i, doc in enumerate(documents[:3]):
        print(f"\nSample Document {i+1}:")
        print(f"  Source: {doc.metadata.get('url', 'Unknown')}")
        print(f"  Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"  Section: {doc.metadata.get('section', 'Unknown')}")
        print(f"  Content Length: {len(doc.page_content)} characters")
        print(f"  Content Preview: {doc.page_content[:150]}...")
    
    print()
    
    # Confirm upload
    response = input("🚀 Do you want to proceed with uploading to Pinecone? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Upload cancelled.")
        return False
    
    print()
    print("🔄 UPLOADING TO PINECONE")
    print("-" * 40)
    
    # Upload to Pinecone
    success = upload_scraped_data_to_pinecone(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        batch_size=50  # Smaller batches for better progress tracking
    )
    
    if success:
        print()
        print("✅ SUCCESS!")
        print(f"Successfully uploaded {len(documents)} documents to Pinecone index '{pinecone_index_name}'")
        print()
        print("📋 NEXT STEPS:")
        print("1. Test the chatbot with PEC-related questions")
        print("2. Verify responses include scraped web content")
        print("3. Check conversation logs for quality")
        print()
        print("🧪 TEST QUESTIONS TO TRY:")
        print("- What are the requirements for Professional Engineer registration?")
        print("- How do I register my engineering firm?")
        print("- What is Continuing Professional Development?")
        print("- How to apply for PEC number?")
        print("- What are the accreditation procedures?")
        
    else:
        print()
        print("❌ FAILED!")
        print("Upload to Pinecone failed. Check the logs above for details.")
        return False
    
    return True


def check_integration_status():
    """Check the current status of scraped data integration."""
    print("🔍 INTEGRATION STATUS CHECK")
    print("-" * 40)
    
    # Check data folder
    data_root = Path(__file__).parent.parent / "data"
    chunks_dir = data_root / "pec" / "chunks"
    
    print(f"Data folder: {data_root}")
    print(f"Chunks folder: {chunks_dir}")
    print(f"Chunks folder exists: {chunks_dir.exists()}")
    
    if chunks_dir.exists():
        chunk_files = list(chunks_dir.glob("*.txt"))
        print(f"Chunk files found: {len(chunk_files)}")
        
        index_file = chunks_dir / "chunk_index.json"
        print(f"Index file exists: {index_file.exists()}")
    
    # Check environment
    load_dotenv()
    print(f"PINECONE_API_KEY set: {bool(os.getenv('PINECONE_API_KEY'))}")
    print(f"PINECONE_INDEX_NAME: {os.getenv('PINECONE_INDEX_NAME', 'Not set')}")
    
    # Get stats
    stats = get_scraped_data_stats()
    if "error" not in stats:
        print(f"Total chunks available: {stats.get('total_chunks', 0)}")
    else:
        print(f"Error getting stats: {stats['error']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate scraped PEC data into Pinecone")
    parser.add_argument("--check", action="store_true", help="Check integration status only")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if args.check:
        check_integration_status()
    else:
        if args.force:
            # Override input function for automated runs
            original_input = input
            input = lambda prompt: "y"
        
        try:
            success = main()
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)