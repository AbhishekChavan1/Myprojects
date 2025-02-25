from flask import Flask, request, jsonify
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import cohere
from langchain.prompts import PromptTemplate
import os

app = Flask(__name__)

docs_folder = "/content"  

if not os.path.exists(docs_folder):
    print(f"Error: Folder '{docs_folder}' not found.")
    exit(1) 

loaders = [PyMuPDFLoader(os.path.join(docs_folder, fn)) for fn in os.listdir(docs_folder) if fn.endswith(".pdf")]
documents = []
for loader in loaders:
    documents.extend(loader.load())
print(f"Loaded {len(documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

api='LLPo5KmMNg5BwzC7Xpouw3NekKUW9kFl5uULqxjn' 
llm = cohere.Cohere(cohere_api_key=api)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        result = qa_chain({"query": query})
        return jsonify({'answer': result['result']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
