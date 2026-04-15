import streamlit as st
from Backend_langgraph import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable": {"thread_id": "thread-001"}}


# use session state to store the chat history so that it can be accessed across different runs of the script
# session state is a dictionary that can store any data and it is persistent across different runs of the script

# so chats are does not dissapper when the user input a new message,
# it will be stored in the session state and can be accessed in the next run of the script
# it dissapper when the user refresh the page because session state is stored in the memory and it is cleared when the page is refreshed

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# store the user message and assistant response in chat history
# {'role': 'user'/'assistant', 'content': 'message content'}

# Display the chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here...")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # response = chatbot.invoke(
    #     {"messages": [HumanMessage(content=user_input)]}, config=CONFIG
    # )
    # ai_message = response["messages"][-1].content
    # st.session_state["message_history"].append(
    #     {"role": "assistant", "content": ai_message}
    # )
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
