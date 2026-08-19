import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

prompt = PromptTemplate.from_template("{question}")

model = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=API_KEY)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of India?"})

print(result)
