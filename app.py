import streamlit as st
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

# Load environment variables
load_dotenv()

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
        file_content = read_file_with_fallback_encoding(text_file_path)
        document = Document(page_content=file_content, metadata={"source": text_file_path})
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        splits = text_splitter.split_documents([document])
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type='retrieval_query')
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        return retriever
    except Exception as e:
        st.error(f"Error processing file {text_file_path}: {str(e)}")
        return None

# Initialize Streamlit app
st.title("RAG-based Chatbot")

# File uploader for the text file
uploaded_file = st.file_uploader("Choose a text file for RAG", type="txt")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_rag_file.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load the RAG content
    retriever = load_rag_content_from_text("temp_rag_file.txt")
    
    if retriever:
        # Initialize the LLM
        llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

        # Template for the prompt
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

        # Function to get chatbot response
        def get_chatbot_response(query):
            result = qa_chain({"query": query})
            return result['result'], result['source_documents']

        # Chat interface
        st.subheader("Chat with the RAG-based Chatbot")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is your question?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response, source_documents = get_chatbot_response(prompt)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
                if source_documents:
                    st.markdown("**Sources:**")
                    for doc in source_documents:
                        st.markdown(f"- {doc.metadata['source']}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.error("Failed to process the uploaded file. Please try again.")

    # Clean up the temporary file
    os.remove("temp_rag_file.txt")

else:
    st.info("Please upload a text file to start the chat.")