##PHASE 2

from dotenv import load_dotenv
load_dotenv()
import os

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Gemini LLM integration ---
import google.generativeai as genai

# step1: setup llm (Gemini)
def load_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    return model

# step2: connect llm to FAISS and create chain

DB_FAISS_PATH = "vectorstore/db_faiss/"

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer,just say you dont know,dont try to make up an answer.
Dont provide anything out of the given context.
Context:{context}
Question:{question}
Start the answer directly.No small talk please.
"""

def set_customprompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables={"context", "question"})
    return prompt

# load database
embedded_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedded_model, allow_dangerous_deserialization=True)

# Gemini wrapper for langchain RetrievalQA
class GeminiLLMWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt, **kwargs):
        response = self.model.generate_content(prompt)
        return response.text

# create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=GeminiLLMWrapper(load_llm()),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_customprompt(custom_prompt_template)}
)

# invoke with a single query
user_query = input("Write query Here:")
response = qa_chain.invoke({'query': user_query})
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])
