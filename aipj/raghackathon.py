from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()
api_key=os.getenv("MISTRAL_API_KEY")
file_path = "asn5.pdf"
loader = PyPDFLoader(file_path)

loader = PyPDFLoader(file_path)
pages = loader.load_and_split()
docs = loader.load()

text_splitter=RecursiveCharacterTextSplitter()
documents=text_splitter.split_documents(docs)
embeddings=MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
vector=FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

docs = vector.similarity_search("What are the task?", k=1)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

model = ChatMistralAI(mistral_api_key=api_key)
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")

document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "What were the two main things the author worked on before college?"})
print(response["answer"])