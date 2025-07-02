from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# --- Use LangChain's Gemini integration to use gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI

DB_FAISS_PATH = "vectorstore/db_faiss/"

@st.cache_resource
def get_vectorstore():
    embedded_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedded_model, allow_dangerous_deserialization=True)
    return db

def set_customprompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        google_api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True,
    )
    return llm

def format_sources(source_documents, max_sources=3):
    seen = set()
    count = 0
    for i, doc in enumerate(source_documents):
        src = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        snippet = doc.page_content[:350].replace('\n', ' ') + "..."
        with st.expander(f"Source [{count+1}]: {src} (Page {page})"):
            st.write(snippet)
        count += 1
        if count >= max_sources:
            break
    return ""

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
    st.sidebar.markdown("Your medical AI companion, made by\n**Rajangshu Kumar**", unsafe_allow_html=True)
    st.title("🏥 DocBuddy: Your Personalised Medical Bot")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    vectorstore = get_vectorstore()
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
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 15}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_customprompt(custom_prompt_template)}
    )

    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.chat_message("user").markdown(f"💬 **You:** {message['content']}")
        else:
            st.chat_message("assistant").markdown(f"🤖 **DocBuddy:** {message['content']}")

    prompt = st.chat_input("Pass your prompt here!!")
    if prompt is not None:
        st.chat_message('user').markdown(f"💬 **You:** {prompt}")
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
        if prompt.lower().strip() in greetings:
            response_text = "Hello! 👋 How can I assist you with your medical document questions today?"
            st.chat_message('assistant').markdown(f"🤖 **DocBuddy:** {response_text}")
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
        else:
            with st.spinner("DocBuddy is thinking..."):
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]
                st.chat_message('assistant').markdown(f"🤖 **DocBuddy:** {result}")
                st.session_state.messages.append({'role': 'assistant', 'content': result})
            st.markdown("#### 🔗 Top Source Documents")
            format_sources(source_documents, max_sources=3)

if __name__ == "__main__":
    main()
