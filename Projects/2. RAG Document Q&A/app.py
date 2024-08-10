import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
import streamlit as st
from dotenv import load_dotenv
import time
load_dotenv()

# Initializing Environment Variables
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")


# initializing the model
llm=ChatGroq(model="Llama3-8b-8192",groq_api_key=groq_api_key)

# prompt
prompt=ChatPromptTemplate.from_template(
    """Answer the question based on the provided context only. 
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}"""
    )

def create_vector_embeddings(doc):
    
    if "vectors" not in st.session_state:
        # st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.embeddings=OpenAIEmbeddings()
        # data ingestion and loading
        
        # if wanted to load the complete directory
        # st.session_state.loader=PyPDFDirectoryLoader("docs")

        # if wanted to upload the file on streamlit
        if doc is not None:
            with open("temp_uploaded_file.pdf", "wb") as f:
                f.write(doc.read())
        st.session_state.loader=PyPDFLoader("temp_uploaded_file.pdf")
        
        # load the docs
        st.session_state.docs=st.session_state.loader.load()

        # text Splitters
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        # vector store db
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

st.title("RAG Document Q&A using Open AI Embeddings and LLAMA3")

# uploading file on streamlit
uploaded_file = st.file_uploader("Upload Your File", type="pdf")


if st.button("Document Embedding"):
    try:
        # storing the file data into vector db
        create_vector_embeddings(uploaded_file)
        st.write("Your Vector DB is ready!")
    except Exception as e:
        st.write("Upload the correct file")

user_prompt=st.text_input("Enter Your Query from the documents")

if user_prompt:
    # creating a documents chain on llm and prompt
    documents_chain=create_stuff_documents_chain(llm,prompt)
    # creating retriever to retrieve the data from vector db based on the user input
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,documents_chain)
    start=time.process_time()
    # invoking the user input
    response=retrieval_chain.invoke({"input":user_prompt})
    print(f"Response Time: {time.process_time()-start}")
    st.write(response['answer'])

    # getting context from vectordb on which the response was returned  
    with st.expander("document Similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------")