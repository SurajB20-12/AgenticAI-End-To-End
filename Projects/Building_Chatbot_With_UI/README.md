# Building a Chatbot with LangGraph and Streamlit

This project demonstrates the creation of a chatbot application using **LangGraph** as the backend and **Streamlit** for the user interface. The chatbot is designed to provide a seamless conversational experience while leveraging advanced features such as **persistent state management** and **short-term memory**.

---

## Features

### 1. **LangGraph Backend**

- **LangGraph** is used to define and manage the workflow of the chatbot.
- It enables the chatbot to maintain the **entire state of the workflow**, ensuring a consistent and logical flow of conversation.

### 2. **Streamlit Frontend**

- The user interface is built using **Streamlit**, providing an interactive and visually appealing experience.
- Streamlit allows for rapid prototyping and deployment of the chatbot UI.

### 3. **Persistent State Management**

- The chatbot uses the concept of **persistence** to store the entire state of the workflow.
- This ensures that the chatbot can maintain context across multiple interactions, even if the session is interrupted.

### 4. **Short-Term Memory**

- The chatbot is equipped with **short-term memory**, allowing it to remember recent interactions within a session.
- This enhances the conversational experience by making the chatbot more responsive and context-aware.

---

## Project Structure

```
└── 📁Building_Chatbot_With_UI
    └── 📁__pycache__
        ├── Backend_langgraph.cpython-314.pyc
    ├── .env
    ├── Backend_langgraph.py
    ├── Frontend_Steamlit.py
    └── README.md
```

## Installation

To run this project, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repo/Building_Chatbot_With_UI.git
   cd Building_Chatbot_With_UI

   ```

2. **Install Dependencies**
   Ensure you have Python installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt

   ```

3. **Run Application**

   Start the Backend

   ```bash
   python Backend_langgraph.py

   ```

   Start the Backend

   ```bash
   streamlit run Frontend_Steamlit.py

   ```

## Usage

1. Open the Streamlit application in your browser (usually at http://localhost:8501).
2. Interact with the chatbot through the user-friendly interface.
3. Observe how the chatbot maintains context and remembers recent interactions.

## Why This Chatbot is Special

**Persistent Workflow State**: Unlike traditional chatbots, this application stores the entire workflow state, ensuring that the conversation remains consistent and logical.
**Short-Term Memory**: The chatbot remembers recent interactions, making it more intelligent and user-friendly.
**Seamless Integration**: Combines the power of LangGraph for backend logic with Streamlit for an intuitive frontend.

## Contributing

Contributions are welcome! If you'd like to improve this project, please fork the repository and submit a pull request.
