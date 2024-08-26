import os
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


# Load environment variables (make sure to set GOOGLE_API_KEY in your .env file)
load_dotenv()

def read_file_with_fallback_encoding(file_path):
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file {file_path} with any of the attempted encodings")

def load_rag_content_from_text(text_file_path, chunk_size=1000, chunk_overlap=200):
    try:
        # Read the file content
        file_content = read_file_with_fallback_encoding(text_file_path)
        
        # Create a Document object
        document = Document(page_content=file_content, metadata={"source": text_file_path})

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Split the document
        splits = text_splitter.split_documents([document])

        # Initialize the embedding function with Gemini Pro
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type='retrieval_query')

        # Create the vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)

        # Create the retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        return retriever
    except Exception as e:
        print(f"Error processing file {text_file_path}: {str(e)}")
        return None

# Usage
text_file_path = "rag_string.txt"
retriever = load_rag_content_from_text(text_file_path)

def RAGbasedChatbot(text_file_path):

    # Initialize the LLM (assuming you have the Gemini model set up)
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7) 

    # Template to format the prompt for Gemini, incorporating the retrieved context
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

    # Chain to combine the retriever and the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # Function to get user input and generate a response
    def get_chatbot_response(query):
        # Get relevant context from the retriever
        result = qa_chain({"query": query})

        # Return the answer and the source documents used for generating the answer
        return result['result'], result['source_documents'] 

    # Chatbot loop
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        response, source_documents = get_chatbot_response(query)
        print("Question:", query)
        print("Answer:", response)
        
    # # Optionally, print the source documents used for generating the response
    # if source_documents:
    #     print("\nSources:")
    #     for doc in source_documents:
    #         print(doc.metadata['source'])
    
RAGbasedChatbot(text_file_path)