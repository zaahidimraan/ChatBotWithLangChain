import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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
    
    # Initialize the LLM
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=utils.get_api_key("GOOGLE_API_KEY"))
    
    condense_question_system_template = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_question_prompt
    )

    system_prompt = (
    """You are an Education Consultant at Routes Overseas Consultants. Your job is to provide information to students about Routes Overseas Consultants, IELTS, PTE and queries about studies in UK, Australia, Canada and New zeeland.
        Guidelines:
        1. Answer the queries using context about Routes Overseas Consultants, IELTS, PTE and queries about studies in UK, Australia, Canada and New zeeland.
        2. Primary use context to answer the queries and if the context does not have relevant information to the query, you can use your own knowledge to answer the query but the answer should be relevant and accurate to the Routes Overseas Consultants, IELTS, PTE and studies in UK, Australia, Canada and New zeeland.
        3. If you are not sure about the answer, you can ask the student to visit the website of Routes Overseas Consultants[http://www.rosconsultants.com/] for more information or contact them on this number: 0321 7758462.
        4. If you do not understand the query, you can ask the student to rephrase the query.
        5. If user ask cost related query without giving any detail, provide on average cost from Pakistan and ask user to provide more details to get personalized answer.
        6. Do not answer to general questions that is not related to Routes Overseas Consultants, IELTS, PTE or studies in UK, Australia, Canada and New zeeland.  
        7. If user ask question about other consultants, remind them that you are a consultant at Routes Overseas Consultants and you can only provide information about Routes Overseas Consultants.
        8. Use emojis sparingly in response to make the conversation more engaging.
        \n\n
        Context: {context}"""
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Store variables in session state
    st.session_state.qa_chain = convo_qa_chain
    return True

# Initialize Streamlit app
st.title("Routes Overseas Consultants Chatbot")

# Initialize session state variables only once
if "retriever" not in st.session_state:
    if not initialize_session_state():
        st.stop()  # Stop app if initialization failed
        
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Assign Description to website
response = utils.get_chatbot_response(f"Give only the brief description of Routes Overseas Consultants to set as chatbot description.",st.session_state.qa_chain,st.session_state.messages)
# Chat interface
st.subheader(response)


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Get the response from the qa_chain
    response= utils.get_chatbot_response(prompt,st.session_state.qa_chain,st.session_state.messages)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
