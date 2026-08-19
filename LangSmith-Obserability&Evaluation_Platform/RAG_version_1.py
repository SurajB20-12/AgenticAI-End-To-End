import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

API_KEY = os.getenv("API_KEY")

PDF_PATH = "ML_book.pdf"

# Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

raw_splits = splitter.split_documents(docs)

splits = []

for doc in raw_splits:
    # Force cast page_content to string and strip spaces
    clean_text = str(doc.page_content).strip()

    # Strictly ensure it is a valid, non-empty text string
    if clean_text and clean_text != "None" and len(clean_text) > 0:
        doc.page_content = clean_text  # Reassign the sanitized string
        splits.append(doc)


# HuggingFace Embeddings
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Create FAISS index
vs = FAISS.from_documents(documents=splits, embedding=emb)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY from the provided context. If not found, say you don't know.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

# LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=API_KEY, temperature=0.7)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


parallel = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

chain = parallel | prompt | llm | StrOutputParser()

print("PDF RAG Ready!")

question = input("\nQuestion: ")

answer = chain.invoke(question)

print("\nAnswer:\n")
print(answer)
