##PHASE 1
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


## step 1:load the raw pdf

##basically this func loads all the pdf pages together
DATA_PATH="data/"
def load_pdf_files(data):
    loader=DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
#print("length of doc:",len(documents))

## step 2: create chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(documents)
#print("length of chunks:",len(text_chunks))

##vector embeddings of created chunks
def get_embedding_model():
    embedded_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedded_model

embedded_model=get_embedding_model()

##step 4: store embeddings in faiss for semantic search
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedded_model)
db.save_local(DB_FAISS_PATH)

