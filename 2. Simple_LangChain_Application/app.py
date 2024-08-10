import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()


# Importing API Keys into environment

# Langchain api key for langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# Open AI Credentials from .env file
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="SIMPLE GEN AI APPLICATION"

# prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistant. please respond to the question asked."),
        ("user","Question:{question}")
    ]
    )

st.title("Langchain Demo with Open AI GPT-4")

input_text=st.text_input("What question you have in the mind?")

# Open AI GPT-4 model

llm=ChatOpenAI(model="gpt-4",temperature=0.3)

ouput_parser=StrOutputParser()

chain=prompt|llm|ouput_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))