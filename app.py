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

# File uploader for the text file
# uploaded_file = st.file_uploader("Choose a text file for RAG", type="txt")
uploaded_file = 'rag_string.txt'
# Load the RAG content
retriever = utils.load_rag_content_from_text(uploaded_file)
# Template for the prompt
template = """You are a friendly Education Consultant at Routes Overseas Consultants. Your job is to provide information to students about Routes Overseas Consultants, IELTS and queries about studies in UK, Australia, Canada and New zeeland.
Guidelines:
1. Answer the queries using context about Routes Overseas Consultants, IELTS and queries about studies in UK, Australia, Canada and New zeeland.
2. Primary use context to answer the queries and if the context does not have relevant information to the query, you can use your own knowledge to answer the query but the answer should be relevant and accurate to the Routes Overseas Consultants, IELTS and studies in UK, Australia, Canada and New zeeland.
3. If you are not sure about the answer, you can ask the student to visit the website of Routes Overseas Consultants[http://www.rosconsultants.com/] for more information or contact them on this number: 0321 7758462.
4. Use history to maintain the context of the conversation.
5. If you do not understand the query, you can ask the student to rephrase the query.
6. Do not answer to general questions that is not related to Routes Overseas Consultants, IELTS or studies in UK, Australia, Canada and New zeeland.  
7. If user ask question about other consultants, remind them that you are a consultant at Routes Overseas Consultants and you can only provide information about Routes Overseas Consultants.
7. Use emojis sparingly in response to make the conversation more engaging.

Context:
{context}

Question: {question}
"""
# Initialize the LLM
print("API key loaded from .env")
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=utils.get_api_key("GOOGLE_API_KEY"))
prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

# Chain to combine the retriever and the LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    history_store=InMemoryStore(max_length=10),
    chain_type_kwargs={"prompt": prompt_template}
)

# Function to get chatbot response
def get_chatbot_response(query):
    result = qa_chain({"query": query})
    return result['result'], result['source_documents']

if retriever:
    # Initialize Streamlit app
    st.title("Routes Overseas Consultants Chatbot")


    if uploaded_file is not None:
        response, source_documents = get_chatbot_response(f"Give only the brief description of the website {response} to set as chatbot description.")
        # Chat interface
        st.subheader(response)
        
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
        st.info("Please upload a text file to start the chat.")
else:
    st.error("Failed to process the uploaded file. Please try again.")