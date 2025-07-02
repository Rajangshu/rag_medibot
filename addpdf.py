import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss/"

# 1. Load only the PDFs currently in data/
loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# 3. Load existing vectorstore
embedded_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedded_model, allow_dangerous_deserialization=True)

# 4. Add new chunks to the vectorstore
db.add_documents(text_chunks)

# 5. Save the updated vectorstore
db.save_local(DB_FAISS_PATH)

print("Successfully added the new PDFs to the FAISS vectorstore!")
