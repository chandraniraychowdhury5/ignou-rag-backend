# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os

from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ========== APP INIT ==========
app = FastAPI()

# Enable CORS (for Netlify frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== GLOBAL VECTOR STORE ==========
vector_db = None

# ========== HOME ROUTE ==========
@app.get("/")
def home():
    return {"message": "AI PDF QA API running successfully on Render 🚀"}

# ========== UPLOAD PDF ==========
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_db

    try:
        # Save file
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = splitter.split_documents(documents)

        # Embeddings (lightweight & stable)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create FAISS vector store
        vector_db = FAISS.from_documents(texts, embeddings)

        return {"message": "PDF uploaded & processed successfully ✅"}

    except Exception as e:
        return {"error": str(e)}

# ========== ASK QUESTION ==========
@app.get("/ask/")
def ask_question(query: str):
    global vector_db

    if vector_db is None:
        return {"error": "Please upload a PDF first"}

    try:
        docs = vector_db.similarity_search(query, k=3)

        answer = "\n\n".join([doc.page_content for doc in docs])

        return {
            "query": query,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}

# ========== RENDER PORT FIX ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # IMPORTANT
    uvicorn.run(app, host="0.0.0.0", port=port)
