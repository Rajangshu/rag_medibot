# connect_llm.py

from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_core.documents import Document

# --- Load Gemini LLM ---
def load_llm():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel("gemini-2.0-flash-001")

# --- Set custom prompt ---
custom_prompt_template = """
You are a highly knowledgeable and detail-oriented medical assistant.
Using ONLY the information provided in the context below, answer the user's question as thoroughly and precisely as possible.
Your response must:
- Be **accurate, comprehensive, very detailed**, and based solely on the context — do not infer, guess, or add information not supported by the context.
- Use **clear and precise medical terminology** appropriate for an informed but non-expert audience.
- Provide **explanations, relevant pathophysiology, examples, and highlight exceptions or uncommon presentations** when applicable.
- Organize your answer using **bullet points** and **bold key terms** for clarity and easy reading.
- Use **inline citation numbers** like [1], [2], etc., immediately following factual statements.
- **Group related information logically** to avoid redundancy and improve the flow of the answer.
- For timing- or onset-related questions, clearly specify if the symptom or event is **acute, subacute, or chronic**, including specific timeframes if available.
- If the answer cannot be found in the context, reply with **"I don't know"** without speculation or fabrication.
- Conclude with a **concise practical takeaway or recommendation**, when relevant.
Maintain a professional, patient-friendly, and informative tone throughout.
---
Context:
{context}
Question:
{question}
---
Provide a clear, well-structured, and well-supported response:
"""

def set_customprompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --- Pinecone v3 Setup ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- Embedder ---
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Custom Retriever ---
class PineconeRetriever:
    def __init__(self, index, embedder, namespace=None):
        self.index = index
        self.embedder = embedder
        self.namespace = namespace

    def get_relevant_documents(self, query):
        embedding = self.embedder.embed_query(query)
        results = self.index.query(vector=embedding, top_k=8, include_metadata=True, namespace=self.namespace)

        documents = []
        for match in results['matches']:
            metadata = match['metadata']
            text = metadata.get('text', '')
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        return documents

    def as_retriever(self, search_kwargs=None):
        return self

# --- LLM Wrapper for LangChain ---
class GeminiLLMWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt, **kwargs):
        response = self.model.generate_content(prompt)
        return response.text

# --- Create QA Chain ---
retriever = PineconeRetriever(index=index, embedder=embedder)

qa_chain = RetrievalQA.from_chain_type(
    llm=GeminiLLMWrapper(load_llm()),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_customprompt(custom_prompt_template)}
)

# --- Run Interaction ---
user_query = input("💬 Ask a medical question: ")
response = qa_chain.invoke({'query': user_query})

print("\n🤖 RESULT:\n", response["result"])
print("\n📚 SOURCE DOCUMENTS:\n")
for i, doc in enumerate(response["source_documents"], 1):
    print(f"[{i}] From page {doc.metadata.get('page', 'N/A')} in {doc.metadata.get('source', 'Unknown')}")
    print(doc.page_content[:400] + "...\n")
