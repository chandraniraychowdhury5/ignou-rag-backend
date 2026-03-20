from fastapi import FastAPI
from pydantic import BaseModel
import requests
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader

app = FastAPI()

PDF_URL = "PUT_YOUR_SUPABASE_PDF_URL"

# Download PDF
import os
if not os.path.exists("file.pdf"):
    r = requests.get(PDF_URL)
    open("file.pdf", "wb").write(r.content)

# Load PDF
loader = PyPDFLoader("file.pdf")
documents = loader.load()

# Split
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embedding
embeddings = HuggingFaceEmbeddings()

# Vector DB
db = FAISS.from_documents(docs, embeddings)

# Local LLM
pipe = pipeline("text-generation", model="distilgpt2")
llm = HuggingFacePipeline(pipeline=pipe)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG Running"}

@app.post("/ask")
def ask(q: Query):
    return {"answer": qa.run(q.question)}
