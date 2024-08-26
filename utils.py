
import os
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.storage import InMemoryStore
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Function to get API key from Streamlit secrets or environment variable
def get_api_key(key_name):
    if key_name not in os.environ:
        print(f"Key {key_name} not found in environment variables, checking Streamlit secrets")
        print(st.secrets.get(key_name))
        return st.secrets.get(key_name)
    else:
        return os.getenv(key_name)
# Function to read file with fallback encoding
def read_file_with_fallback_encoding(file_path):
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file {file_path} with any of the attempted encodings")

# Function to load RAG content from text
def load_rag_content_from_text(text_file_path, chunk_size=1000, chunk_overlap=200):
    try:
        st.info(f"Processing file {text_file_path}")
        file_content = read_file_with_fallback_encoding(text_file_path)
        document = Document(page_content=file_content, metadata={"source": text_file_path})
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        splits = text_splitter.split_documents([document])
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type='retrieval_query', google_api_key=get_api_key("GOOGLE_API_KEY"))
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        return retriever
    except Exception as e:
        st.error(f"Error processing file {text_file_path}: {str(e)}")
        return None