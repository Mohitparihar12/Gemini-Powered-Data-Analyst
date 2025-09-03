import sys
import os
import asyncio
import nest_asyncio
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 🛠 Ensure event loop works with Streamlit + gRPC
nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 🔐 Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ GOOGLE_API_KEY not found in .env file")
    st.stop()

# ✅ Configure genai
genai.configure(api_key=api_key)

# ✅ Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# 📄 Load PDF
pdf_path = r"C:\Python for Data Analytics\dataset\Datasets\PROJECTS\my_paper.pdf"
loader = PyPDFLoader(pdf_path)
data = loader.load()

# ✂ Split text into smaller chunks to avoid embedding errors
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

# 🧠 Create vectorstore
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 🤖 Define Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.3
)

# 💬 Streamlit UI
st.title("📑 PDF Chat with Gemini + Chroma")

query = st.chat_input("Ask me anything about the PDF:")

# 📝 Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for answering questions from PDF documents. Use the provided context: {context}"),
    ("human", "{input}")
])

# 🔄 Run RAG pipeline
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        response = rag_chain.invoke({"input": query})
        st.write("### Answer:")
        st.write(response["answer"])
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
