from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser
import streamlit as st
import os

os.environ["OPENAI_API_KEY"]=""
os.environ["LANGCHAIN_TRACING_V2"]="true"

prompt_template=ChatPromptTemplate.from_messages([("system","You are a helpful assistant. Please respond to the user queries"),
                                                  ("user","Question: {question}")])

# Streamlit

st.title("Langchain Chatbot Assistant")
input_text=st.text_input("Ask a Question")

llm=OpenAI(model="gpt-3.5-turbo")
# output_parser=StructuredOutputParser()

chains=prompt_template|llm

if input_text:
  st.write(chains.invoke({"question":input_text}))
# exec(code, locals())