# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
Common Index Builder for RAG Examples
Creates FAISS vector index from product documentation that all framework examples can share.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

def main():
    print("🔨 Building FAISS Index for Product Documentation\n")
    print("=" * 60)

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(script_dir, "sample-docs")
    index_path = os.path.join(script_dir, "faiss_index")

    # Load documents
    print("📄 Loading documents...")
    loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"   Loaded {len(documents)} documents")

    # Split documents
    print("✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"   Created {len(splits)} chunks")

    # Create embeddings and index
    print("🔮 Creating embeddings and building index...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Save index
    print("💾 Saving index to disk...")
    vectorstore.save_local(index_path)

    print("=" * 60)
    print(f"✅ Index built successfully!")
    print(f"   Location: {index_path}")
    print(f"   Documents: {len(documents)}")
    print(f"   Chunks: {len(splits)}")
    print("\nAll framework examples can now load this pre-built index.")

if __name__ == "__main__":
    main()
