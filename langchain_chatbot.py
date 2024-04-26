# from langchain.llms import OpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.output_parsers import StructuredOutputParser
# import streamlit as st
# import os

from langchain.llms import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
# from dotenv import load_dotenv,find_dotenv
# load_dotenv(find_dotenv())

os.environ["OPENAI_API_KEY"]=""
## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=""
prompt_template=ChatPromptTemplate.from_messages([("system","You are a helpful assistant. Please respond to the user queries"),
                                                  ("user","Question: {question}")])

# Streamlit

st.title("Langchain Chatbot Assistant")
input_text=st.text_input("Ask a Question")

llm=OpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()

chains=prompt_template|llm|output_parser

if input_text:
  st.write(chains.invoke({"question":input_text}))
# exec(code, locals())
