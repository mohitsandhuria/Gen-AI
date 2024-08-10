from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langserve import add_routes

# loading the Groq API Key
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")

# Initializing the Gemma2 Model with 9 billion parameter
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

# create prompt Template
generic_template="Translate the following into {language}"
prompt=ChatPromptTemplate.from_messages(
    [("system",generic_template),
     ("user","{text}")]
    )

# create output parser for formatting the output
parser=StrOutputParser()

# creating chain
chain= prompt|model|parser

# App definition

app=FastAPI(title="Langchain Server",
            version="1.0",
            description="This is simple app created using REST API and Langserve")

add_routes(app,chain,path="/chain")

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)