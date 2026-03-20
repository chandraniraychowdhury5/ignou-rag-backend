from fastapi import FastAPI
from pydantic import BaseModel
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader

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
