import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pec-assistant-index")

UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

# Check index configuration
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # For GoogleGenerativeAI embeddings
        metric="cosine",  # Changed to cosine to match your index
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)
    print(f"✅ Index {PINECONE_INDEX_NAME} created successfully")
else:
    print(f"Using existing index: {PINECONE_INDEX_NAME}")
    # Verify dimensions match
    index_info = pc.describe_index(PINECONE_INDEX_NAME)
    if hasattr(index_info, 'dimension') and index_info.dimension != 384:
        print(f"⚠️  WARNING: Index dimension is {index_info.dimension}, but HuggingFace embeddings are 384")
        print(f"⚠️  You need to either:")
        print(f"   1. Delete the index and recreate it with dimension 384")
        print(f"   2. Or use embeddings that match dimension {index_info.dimension}")
        raise ValueError(f"Dimension mismatch: Index={index_info.dimension}, Embeddings=384")

index = pc.Index(PINECONE_INDEX_NAME)


def get_file_loader(file_path):
    """
    Return appropriate loader based on file extension.
    Supports both PDF and TXT files.
    """
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension == '.txt':
        return TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Only .pdf and .txt are supported.")


def load_vectorstore(uploaded_files, index_name=None):
    """
    Load, split, embed and upsert file content to Pinecone.
    
    Args:
        uploaded_files: List of uploaded file objects
        index_name: Optional custom index name (overrides env variable)
    """
    # Use custom index name if provided
    target_index_name = index_name or PINECONE_INDEX_NAME
    target_index = pc.Index(target_index_name)
    
    # Use HuggingFace embeddings (free, runs locally, no API limits)
    print("Loading embedding model (first run downloads model, ~90MB)...")
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast & accurate
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✓ Embedding model loaded")
    file_paths = []
    
    print(f"\n{'='*80}")
    print(f"📁 Processing {len(uploaded_files)} file(s)")
    print(f"🎯 Target Index: {target_index_name}")
    print(f"{'='*80}\n")

    # Save uploaded files
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))
        print(f"💾 Saved: {file.filename}")

    total_chunks_uploaded = 0
    
    # Process each file
    for file_path in file_paths:
        try:
            file_name = Path(file_path).name
            file_extension = Path(file_path).suffix.lower()
            
            print(f"\n📄 Processing: {file_name} ({file_extension})")
            
            # Load document using appropriate loader
            loader = get_file_loader(file_path)
            documents = loader.load()
            print(f"  ✓ Loaded {len(documents)} document(s)")

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Increased for better context
                chunk_overlap=200  # Increased overlap for continuity
            )
            chunks = splitter.split_documents(documents)
            print(f"  ✓ Split into {len(chunks)} chunks")

            # Prepare data for embedding
            texts = [chunk.page_content for chunk in chunks]
            
            # Enhanced metadata with source information
            metadatas = []
            for i, chunk in enumerate(chunks):
                meta = chunk.metadata.copy()
                meta['source_file'] = file_name
                meta['file_type'] = file_extension.replace('.', '')
                meta['chunk_index'] = i
                meta['total_chunks'] = len(chunks)
                metadatas.append(meta)
            
            # Generate unique IDs
            file_stem = Path(file_path).stem
            ids = [f"{file_stem}-chunk-{i}" for i in range(len(chunks))]

            # Generate embeddings
            print(f"  🔍 Generating embeddings for {len(texts)} chunks...")
            embeddings = embed_model.embed_documents(texts)
            print(f"  ✓ Embeddings generated")

            # Upsert to Pinecone in batches
            batch_size = 100
            print(f"  📤 Uploading to Pinecone (batch size: {batch_size})...")
            
            with tqdm(total=len(embeddings), desc=f"  Uploading {file_name}") as progress:
                for i in range(0, len(embeddings), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_embeddings = embeddings[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    
                    # Create vectors with metadata
                    vectors = [
                        {
                            'id': vid,
                            'values': emb,
                            'metadata': {**meta, 'text': texts[i+j]}
                        }
                        for j, (vid, emb, meta) in enumerate(
                            zip(batch_ids, batch_embeddings, batch_metadatas)
                        )
                    ]
                    
                    target_index.upsert(vectors=vectors)
                    progress.update(len(batch_ids))
            
            total_chunks_uploaded += len(chunks)
            print(f"  ✅ Upload complete for {file_name}")
            
        except Exception as e:
            print(f"  ❌ Error processing {file_name}: {str(e)}")
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ UPLOAD COMPLETE")
    print(f"Total chunks uploaded: {total_chunks_uploaded}")
    print(f"Index: {target_index_name}")
    
    # Get index stats
    try:
        stats = target_index.describe_index_stats()
        print(f"Total vectors in index: {stats.total_vector_count}")
    except Exception as e:
        print(f"Could not fetch index stats: {e}")
    
    print(f"{'='*80}\n")


def get_index_stats(index_name=None):
    """
    Get statistics about the Pinecone index.
    """
    target_index_name = index_name or PINECONE_INDEX_NAME
    target_index = pc.Index(target_index_name)
    
    try:
        stats = target_index.describe_index_stats()
        print(f"\n{'='*80}")
        print(f"INDEX STATISTICS: {target_index_name}")
        print(f"{'='*80}")
        print(f"Total vectors: {stats.total_vector_count}")
        print(f"Dimension: {stats.dimension}")
        if hasattr(stats, 'namespaces'):
            print(f"Namespaces: {stats.namespaces}")
        print(f"{'='*80}\n")
        return stats
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return None


# Optional: Function to delete all vectors from index
def clear_index(index_name=None):
    """
    Clear all vectors from the index. Use with caution!
    """
    target_index_name = index_name or PINECONE_INDEX_NAME
    target_index = pc.Index(target_index_name)
    
    confirm = input(f"⚠️  Are you sure you want to clear all data from '{target_index_name}'? (yes/no): ")
    if confirm.lower() == 'yes':
        target_index.delete(delete_all=True)
        print(f"✅ Index '{target_index_name}' cleared successfully")
    else:
        print("❌ Operation cancelled")


if __name__ == "__main__":
    # Test the configuration
    print(f"\n{'='*80}")
    print("PINECONE CONFIGURATION")
    print(f"{'='*80}")
    print(f"Index Name: {PINECONE_INDEX_NAME}")
    print(f"Region: {PINECONE_ENV}")
    print(f"Upload Directory: {UPLOAD_DIR}")
    print(f"{'='*80}\n")
    
    # Show current index stats
    get_index_stats()
