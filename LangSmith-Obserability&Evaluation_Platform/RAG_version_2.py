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

from langsmith import traceable


load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "RAG_full_pipeline"

API_KEY = os.getenv("API_KEY")

pdf_path = "ML_book.pdf"


@traceable(name="load_pdf")
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()


@traceable(name="split_documents")
def split_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(docs)


@traceable(name="create_vectorstore")
def create_vectorstore(splits):
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(documents=splits, embedding=emb)


@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path, chunk_size=1000, chunk_overlap=150):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = create_vectorstore(splits)

    return vs


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


@traceable(name="rag_pipeline")
def setup_pipeline_and_query(pdf_path, question: str):
    vectorstore = setup_pipeline(pdf_path, chunk_size=1000, chunk_overlap=150)

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    chain = parallel | prompt | llm | StrOutputParser()

    # This LangChain run stays under the same root (since we're inside this traced function)
    lc_config = {"run_name": "pdf_rag_query"}
    return chain.invoke(question, config=lc_config)


if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(pdf_path, q)
    print("\nA:", ans)
