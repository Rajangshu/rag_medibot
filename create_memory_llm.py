import os
import glob
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# Load .env
load_dotenv()
DATA_PATH = "data/"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
API_KEY = os.getenv("PINECONE_API_KEY")
REGION = os.getenv("PINECONE_ENV", "us-east-1")

# Init Pinecone v3
pc = Pinecone(api_key=API_KEY)

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )
index = pc.Index(INDEX_NAME)

# Load PDFs
def load_pdf_files(data_path):
    documents = []
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    for path in pdf_files:
        try:
            print(f"📄 Loading: {path}")
            loader = PyPDFLoader(path)
            docs = loader.load()
            print(f"✅ Loaded {len(docs)} chunks from {path}")
            documents.extend(docs)
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
    return documents

# Load and split
documents = load_pdf_files(DATA_PATH)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embed using HuggingFace
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Prepare for Pinecone upsert
vectors = []
batch_size = 100
for i, doc in enumerate(chunks):
    vector = embedder.embed_query(doc.page_content)
    vectors.append({
        "id": str(uuid.uuid4()),
        "values": vector,
        "metadata": {
            "page_content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", -1),
        }
    })

    # Upload in batches
    if len(vectors) >= batch_size or i == len(chunks) - 1:
        index.upsert(vectors=vectors, namespace="__default__")
        print(f"✅ Uploaded batch {i+1}/{len(chunks)}")
        vectors = []

print("🎉 All chunks uploaded successfully to Pinecone v3!")
