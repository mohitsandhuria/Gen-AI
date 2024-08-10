import streamlit as st
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Intitalizing Environment Variables
# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot With OPEN AI"

# Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [("system","You are a helpful assistant. Please respond to the user queries beyond knowledge."),
     ("user","Question: {question}")]
)

def generate_response(question,api_key,llm,temperature,max_tokens):

    # Initialize Open AI API Key
    openai.api_key=api_key

    # Create Open AI LLM Model by passing parameters
    llm=ChatOpenAI(model=llm,temperature=temperature,max_tokens=max_tokens)

    # Initializing Output parser to get the formatting correctly.
    parser=StrOutputParser()

    #Creating Chain
    chain=prompt|llm|parser

    answer=chain.invoke({"question":question})

    return answer

# creating title
st.title("Q&A Chatbot")

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter the API Key", type="password")
llm=st.sidebar.selectbox("Select an OPEN AI Model",["gpt-4","gpt-4-turbo","gpt-4o"])
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=200)

# Main Interface

st.write("Ask any Question")

user_input=st.text_input("You:")

if user_input:
    response=generate_response(question=user_input,api_key=api_key,llm=llm,temperature=temperature,max_tokens=max_tokens)
    st.write(response)
else:
    st.write("Please Ask Question If you have any for me.")