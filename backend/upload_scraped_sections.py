#!/usr/bin/env python3
"""
Upload scraped PEC sections to Pinecone
Data source: project/data/scraped_data/pec/
Sections: accreditation, cpd, engineers_registration, firm_registration, general
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.load_scraped_sections import (
    get_scraped_sections_stats,
    create_documents_from_scraped_sections,
    upload_scraped_sections_to_pinecone
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to upload scraped sections."""
    print("=" * 80)
    print("UPLOAD SCRAPED PEC SECTIONS TO PINECONE")
    print("Source: data/scraped_data/pec/")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "pec-assistant-index")
    
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY not found in environment variables")
        return False
    
    print(f"Pinecone Index: {pinecone_index_name}")
    print(f"API Key: {'*' * (len(pinecone_api_key) - 4) + pinecone_api_key[-4:]}")
    print()
    
    # Get statistics
    print("📊 SCRAPED SECTIONS STATISTICS")
    print("-" * 40)
    stats = get_scraped_sections_stats()
    
    if "error" in stats:
        logger.error(f"Error: {stats['error']}")
        return False
    
    print(f"Total files: {stats.get('total_files', 0)}")
    print(f"\nSections found:")
    for section, count in stats.get('sections', {}).items():
        print(f"  - {section}: {count} files")
    
    print()
    
    # Ask which sections to exclude
    print("📋 SECTION SELECTION")
    print("-" * 40)
    print("The 'general' section may already be uploaded as chunks.")
    print("Do you want to exclude it? (recommended if you already uploaded chunks)")
    print()
    
    exclude_general = input("Exclude 'general' section? (Y/n): ").strip().lower()
    exclude_sections = ['general'] if exclude_general != 'n' else []
    
    print()
    
    # Create documents
    print("📄 LOADING DOCUMENTS")
    print("-" * 40)
    documents = create_documents_from_scraped_sections(exclude_sections=exclude_sections)
    
    if not documents:
        logger.error("No documents to upload")
        return False
    
    # Count by section
    docs_by_section = {}
    for doc in documents:
        section = doc.metadata.get('section_folder', 'unknown')
        docs_by_section[section] = docs_by_section.get(section, 0) + 1
    
    print(f"Total documents: {len(documents)}")
    print(f"\nDocuments by section:")
    for section, count in docs_by_section.items():
        print(f"  ✅ {section}: {count} documents")
    
    if exclude_sections:
        print(f"\n⏭️  Excluded sections: {', '.join(exclude_sections)}")
    
    print()
    
    # Show samples
    print("📄 SAMPLE DOCUMENTS")
    print("-" * 40)
    sections_shown = set()
    for doc in documents[:5]:
        section = doc.metadata.get('section_folder', 'unknown')
        if section not in sections_shown:
            print(f"\n{section}:")
            print(f"  File: {doc.metadata.get('file_name')}")
            print(f"  Length: {len(doc.page_content)} chars")
            print(f"  Preview: {doc.page_content[:100]}...")
            sections_shown.add(section)
    
    print()
    
    # Confirm upload
    response = input(f"🚀 Upload {len(documents)} documents to Pinecone? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Upload cancelled.")
        return False
    
    print()
    print("🔄 UPLOADING TO PINECONE")
    print("-" * 40)
    
    # Upload
    success = upload_scraped_sections_to_pinecone(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        exclude_sections=exclude_sections,
        batch_size=50
    )
    
    if success:
        print()
        print("✅ SUCCESS!")
        print(f"Uploaded {len(documents)} documents from scraped sections")
        print()
        print("📋 UPLOADED SECTIONS:")
        for section, count in docs_by_section.items():
            print(f"  ✅ {section}: {count} documents")
        print()
        print("🧪 TEST WITH THESE QUESTIONS:")
        print("- What are the PEC accreditation requirements?")
        print("- Tell me about CPD for engineers")
        print("- How do I register as a professional engineer?")
        print("- What are firm registration requirements?")
    else:
        print()
        print("❌ UPLOAD FAILED")
        print("Check the logs above for details")
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload scraped PEC sections")
    parser.add_argument("--force", action="store_true", help="Skip prompts")
    parser.add_argument("--include-general", action="store_true", help="Include general section")
    
    args = parser.parse_args()
    
    if args.force:
        original_input = input
        input = lambda prompt: "y" if "Upload" in prompt else ("n" if args.include_general else "y")
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)