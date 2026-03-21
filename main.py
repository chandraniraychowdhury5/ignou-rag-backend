from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ========== CONFIG ==========
PDF_URL = "PUT_YOUR_SUPABASE_PDF_URL_HERE"

# ========== APP INIT ==========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== GLOBAL ==========
vector_db = None

# ========== LOAD PDF ==========
def initialize_db():
    global vector_db

    if vector_db is not None:
        return

    print("Loading PDF...")

    # Download PDF
    r = requests.get(PDF_URL)
    with open("file.pdf", "wb") as f:
        f.write(r.content)

    # Load PDF
    loader = PyPDFLoader("file.pdf")
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vector_db = FAISS.from_documents(texts, embeddings)

    print("Vector DB ready ✅")

# ========== HOME ==========
@app.get("/")
def home():
    return {"message": "PDF QA backend running (no AI model)"}

# ========== ASK ==========
@app.get("/ask/")
def ask_question(query: str):
    global vector_db

    try:
        initialize_db()

        docs = vector_db.similarity_search(query, k=3)

        if not docs:
            return {"answer": "No relevant content found"}

        # Return raw content (no AI)
        answer = "\n\n".join([doc.page_content for doc in docs])

        return {"answer": answer}

    except Exception as e:
        print("ERROR:", e)
        return {"answer": "Error retrieving answer"}
