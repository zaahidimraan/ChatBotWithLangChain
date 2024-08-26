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
import utils 


# Initialize Streamlit app
st.title("JnS Education")

# File uploader for the text file
# uploaded_file = st.file_uploader("Choose a text file for RAG", type="txt")
uploaded_file = 'rag_string.txt'

if uploaded_file is not None:
    
    # Load the RAG content
    retriever = utils.load_rag_content_from_text(uploaded_file)
    
    if retriever:
        # Initialize the LLM
        llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=utils.get_api_key("GOOGLE_API_KEY"))

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
        st.subheader("Chat with the RAG-based Chatbot for JnS Visa Consultancy")
        
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
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.error("Failed to process the uploaded file. Please try again.")

else:
    st.info("Please upload a text file to start the chat.")