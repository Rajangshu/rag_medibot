# addpdf.py

from dotenv import load_dotenv
load_dotenv()

import os
import glob
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Constants
DATA_PATH = "data/"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
API_KEY = os.getenv("PINECONE_API_KEY")

# --- Load PDFs ---
def load_pdf_files(data_path):
    documents = []
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    for path in pdf_files:
        try:
            print(f"📄 Loading: {path}")
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = path
            print(f"✅ Loaded {len(docs)} chunks from {path}")
            documents.extend(docs)
        except Exception as e:
            print(f"❌ Skipping {path} due to error: {e}")
    return documents

documents = load_pdf_files(DATA_PATH)

# --- Chunk the text ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# --- Load embeddings ---
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors = [
    {
        "id": str(uuid.uuid4()),
        "values": embedder.embed_query(chunk.page_content),
        "metadata": {
            "text": chunk.page_content,
            **chunk.metadata
        }
    }
    for chunk in chunks
]

# --- Initialize Pinecone (v3) ---
pc = Pinecone(api_key=API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    print(f"🛠 Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created index: {INDEX_NAME}")
else:
    print(f"ℹ️ Using existing index: {INDEX_NAME}")

# --- Upload to Pinecone ---
index = pc.Index(INDEX_NAME)
batch_size = 100
total = len(vectors)
print("📤 Uploading to Pinecone...")

for i in range(0, total, batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(vectors=batch)
    print(f"✅ Uploaded batch {i // batch_size + 1}/{(total - 1) // batch_size + 1}")

print("🎉 All new PDFs have been added to Pinecone vectorstore.")
