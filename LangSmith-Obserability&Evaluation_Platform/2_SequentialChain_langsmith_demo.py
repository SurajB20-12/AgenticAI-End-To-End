import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

os.environ["LANGCHAIN_PROJECT"] = "LangSmith-SequentialChain-Demo"

load_dotenv()

API_KEY = os.getenv("API_KEY")

prompt1 = PromptTemplate.from_template(template="Generate a detailed report on {topic}")

prompt2 = PromptTemplate.from_template(
    template="Generate 5 pointer summary from  follwing text \n {text}"
)

model1 = ChatGroq(
    model_name="llama-3.3-70b-versatile", api_key=API_KEY, temperature=0.7
)

model2 = ChatGroq(model_name="llama-3.1-8b-instant", api_key=API_KEY, temperature=0.5)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    "run_name": "SequentialChain LangSmith Demo",
    "tags": ["Sequential workflow", "LangSmith", "LangChain", "Groq"],
    "metadata": {
        "model1": "llama-3.3-70b-versatile",
        "model2": "llama-3.1-8b-instant",
        "temperature1": 0.7,
        "temperature2": 0.5,
    },
}

result = chain.invoke({"topic": "Unemployment in India"}, config=config)

print(result)
