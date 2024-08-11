import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
import streamlit as st
from dotenv import load_dotenv
import time
load_dotenv()

# Initializing Environment Variables
# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# groq_api_key=os.getenv("GROQ_API_KEY")

# initializing the model
# llm=ChatGroq(model="Llama3-8b-8192",groq_api_key=groq_api_key)

# set up streamlit

st.title("Conversational RAG with File Uploads and Chat History")
st.write("Upload PDF's and chat with their content!")

api_key=st.text_input("Enter Groq API Key:",type="password")

if api_key:
    selected_llm=st.sidebar.selectbox("Select your Model",["Gemma2-7b-It","Gemma2-9b-It","Llama3-8b-8192","Llama3.1-8b-Instant"])
    llm=ChatGroq(model=selected_llm,groq_api_key=api_key)

    # chat interface 
    session_id=st.text_input("Session ID",value="default_session")

    # Manage Chat history

    if "store" not in st.session_state:
        st.session_state.store={}
    
    upload_file=st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)
    if upload_file:
        documents=[]
        for files in upload_file:
            temppdf=f"./temp"
            with open(temppdf,"wb") as file:
                file.write(files.getvalue())
                file_name=files.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        
        # Split , Create embedding , store the docs into vector db and initialize retriever
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        split=text_splitter.split_documents(documents=documents)
        vector_stores=Chroma.from_documents(split,embedding=embedding)
        retriever=vector_stores.as_retriever()

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. DO NOT answer the question, "
            "just formulate it if needed and otherwise return as it is."
            )
        
        contextualize_system_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_system_prompt)

        ### Answer Question

        system_prompt=(
            "You are an assistant for question-answering tasks. "
            "use the followingpeices of retrieved context to answer the question"
            "If you don't know the answer say that you don't know. use three sentences" 
            " and keep the answer concise"
            "\n\n"
            "{context}"
            )
        qa_prompt= ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        question_answer_retriever_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_retriever_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain= RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history", 
            output_messages_key="answer"
            )
        

        user_input=st.text_input("Ask your Question")
        if user_input:
            sessison_history= get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.write(st.session_state.store)
            st.write("Assistant: ",response["answer"])
            st.write("Chat History", sessison_history.messages)

else:
    st.write("Please Enter the Groq API Key!")