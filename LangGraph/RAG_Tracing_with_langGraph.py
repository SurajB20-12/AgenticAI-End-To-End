import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langgraph.prebuilt import ToolNode, tools_condition

from langsmith import traceable

load_dotenv()


API_KEY = os.getenv("API_KEY")
llm = ChatGroq(model_name="openai/gpt-oss-20b", api_key=API_KEY)

loader = PyPDFLoader("ML_book.pdf")
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vs = FAISS.from_documents(documents=chunks, embedding=emb)


retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)


@tool
@traceable(name="rag_tool")
def rag_tool(query):
    """
    Retrieve relevant information from the pdf document.
    Use this tool when user ask facual/conceptual questions about the document.
    """
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    return {"query": query, "context": context, "metadata": metadata}


tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@traceable(name="chat_node")
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()


result = chatbot.invoke(
    {
        "messages": [
            HumanMessage(content="Using the pdf, explain what is Supervised Learning?")
        ]
    }
)

print(result["messages"][-1].content)
