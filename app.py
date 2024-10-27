import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import utils

# Function to initialize session variables
def initialize_session_state():
    uploaded_file = 'rag_string.txt'
    
    # Load the RAG content
    retriever = utils.load_rag_content_from_text(uploaded_file)
    if not retriever:
        st.error("Failed to load the RAG content. Please contact on +92 310 6584862.")
        return False

    # Template for the prompt
    template = """You are an Education Consultant at Routes Overseas Consultants. Your job is to provide information to students about Routes Overseas Consultants, IELTS, PTE and queries about studies in UK, Australia, Canada and New zeeland.
    Guidelines:
    1. Answer the queries using context about Routes Overseas Consultants, IELTS, PTE and queries about studies in UK, Australia, Canada and New zeeland.
    2. Primary use context to answer the queries and if the context does not have relevant information to the query, you can use your own knowledge to answer the query but the answer should be relevant and accurate to the Routes Overseas Consultants, IELTS, PTE and studies in UK, Australia, Canada and New zeeland.
    3. If you are not sure about the answer, you can ask the student to visit the website of Routes Overseas Consultants[http://www.rosconsultants.com/] for more information or contact them on this number: 0321 7758462.
    4. If you do not understand the query, you can ask the student to rephrase the query.
    5. If user ask cost related query without giving any detail, provide on average cost from Pakistan and ask user to provide more details to get personalized answer.
    6. Do not answer to general questions that is not related to Routes Overseas Consultants, IELTS, PTE or studies in UK, Australia, Canada and New zeeland.  
    7. If user ask question about other consultants, remind them that you are a consultant at Routes Overseas Consultants and you can only provide information about Routes Overseas Consultants.
    8. Use emojis sparingly in response to make the conversation more engaging.

    Context:
    {context}

    Question: {question}
    """
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

    # Initialize the LLM
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=utils.get_api_key("GOOGLE_API_KEY"))

    # Chain to combine the retriever and the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # Store variables in session state
    st.session_state.retriever = retriever
    st.session_state.qa_chain = qa_chain
    st.session_state.prompt_template = prompt_template
    st.session_state.llm = llm
    return True

# Initialize Streamlit app
st.title("Routes Overseas Consultants Chatbot")

# Initialize session state variables only once
if "retriever" not in st.session_state:
    if not initialize_session_state():
        st.stop()  # Stop app if initialization failed
        
# Assign Description to website
response, source_documents = utils.get_chatbot_response(f"Give only the brief description of the website Routes Overseas Consultants to set as chatbot description.",st.session_state.qa_chain)
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

    # Get the response from the qa_chain
    response, source_documents = utils.get_chatbot_response(prompt,st.session_state.qa_chain)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
