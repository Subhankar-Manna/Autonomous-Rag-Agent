from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from pathlib import Path

from app.graph.state import AgentState

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_DB_PATH = BASE_DIR / "faiss_index"

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vectorstore
vectorstore = FAISS.load_local(
    str(VECTOR_DB_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def executor_agent(state: AgentState) -> AgentState:
    print("🚀 EXECUTOR - FAST RAG MODE")

    try:
        query = state.user_query

        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Answer based on context:

Question:
{query}

Context:
{context}
"""

        response = llm.invoke(prompt)
        state.result = response.content

    except Exception as e:
        state.result = f"Error: {str(e)}"

    return state