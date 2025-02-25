import os
import easyocr
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate

# Flask App Setup
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# OCR Reader
reader = easyocr.Reader(["en"])

# LLM API Key (Replace with your API key)
COHERE_API_KEY = "LLPo5KmMNg5BwzC7Xpouw3NekKUW9kFl5uULqxjn"

# Initialize variables for vector database
db = None

# Function to process PDFs and create vector database
def process_pdf(pdf_path):
    global db
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

# Upload PDF API
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".pdf"):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process the uploaded PDF
        process_pdf(file_path)

        return jsonify({"message": "File uploaded and processed successfully"}), 200
    else:
        return jsonify({"error": "Invalid file format. Only PDFs are allowed"}), 400

# Chat API
@app.route("/ask", methods=["POST"])
def ask_question():
    global db
    if db is None:
        return jsonify({"error": "No document processed yet"}), 400
    
    data = request.get_json()
    query = data.get("query")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Set up the QA model
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know.
    
    {context}
    
    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = Cohere(cohere_api_key=COHERE_API_KEY)
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})

    result = qa_chain({"query": query})
    return jsonify({"answer": result["result"]})

if __name__ == "__main__":
    app.run(debug=True)
