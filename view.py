# view.py
import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from pydantic import Field

# Load environment variables
load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
API_KEY = os.getenv("PINECONE_API_KEY")

# ✅ Custom Pinecone retriever for LangChain (Pinecone v3 compatible)
class PineconeRetriever(BaseRetriever):
    index: object = Field()
    embedding: object = Field()
    top_k: int = 15

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embedding.embed_query(query)
        results = self.index.query(vector=query_vector, top_k=self.top_k, include_metadata=True)
        return [
            Document(
                page_content=match["metadata"].get("page_content", ""),
                metadata=match["metadata"]
            )
            for match in results.get("matches", [])
        ]

@st.cache_resource
def get_retriever():
    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(INDEX_NAME)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return PineconeRetriever(index=index, embedding=embeddings)

# ✅ Your updated rich medical assistant prompt
def set_prompt():
    return PromptTemplate(
        template="""
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
        """,
        input_variables=["context", "question"]
    )

def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",  # or "gemini-1.5-pro-latest"
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )

def format_sources(source_documents, max_sources=3):
    seen = set()
    count = 0
    for doc in source_documents:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        snippet = doc.page_content[:300].replace("\n", " ") + "..."
        with st.expander(f"📄 Source {count+1}: {src} — Page {page}"):
            st.write(snippet)
        count += 1
        if count >= max_sources:
            break

def main():
    st.sidebar.title("🩺 DocBuddy")
    st.sidebar.markdown("""
- 🧑⚕️ Ask about symptoms, treatments, or any health topic—DocBuddy is here to help.
- 📖 Every answer is rooted in trusted, up-to-date medical literature.
- 🌈 *Your reliable partner for medical clarity and confidence!*
""")
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.messages = []
    st.sidebar.markdown("---")
    st.sidebar.markdown("Your medical AI companion, made by  \n**Rajangshu Kumar**", unsafe_allow_html=True)

    st.title("🏥 DocBuddy: Your Personalised Medical Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    retriever = get_retriever()
    prompt_template = set_prompt()
    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Ask your medical question...")
    if user_input:
        st.chat_message("user").markdown(f"💬 **You:** {user_input}")
        st.session_state.messages.append({'role': 'user', 'content': user_input})

        greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
        if user_input.lower().strip() in greetings:
            response_text = "Hello! 👋 How can I assist you with your medical document questions today?"
            st.chat_message("assistant").markdown(f"🤖 **DocBuddy:** {response_text}")
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
        else:
            with st.spinner("DocBuddy is thinking..."):
                response = qa_chain.invoke({'query': user_input})
                result = response["result"]
                source_documents = response["source_documents"]
                st.chat_message("assistant").markdown(f"🤖 **DocBuddy:** {result}")
                st.session_state.messages.append({'role': 'assistant', 'content': result})
                st.markdown("#### 🔗 Top Source Documents")
                format_sources(source_documents, max_sources=3)

if __name__ == "__main__":
    main()
