from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from pathlib import Path

from app.graph.state import AgentState

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_DB_PATH = BASE_DIR / "rag_db"

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if VECTOR_DB_PATH.exists():
    print("FAISS LOADING...")
    vectorstore = FAISS.load_local(
        str(VECTOR_DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("FAISS READY")
else:
    print("FAISS NOT FOUND")
    retriever = None


def executor_agent(state: AgentState) -> AgentState:
    print("🚀 EXECUTOR - FAST RAG MODE")

    try:
        query = state.user_query

        print("STEP 1: Got query")
        if retriever:
            docs = retriever.invoke(query)
            print("STEP 2: Retrieved docs")
        else:
            docs = []
            print("STEP 2: No retriever")
        context = "\n\n".join([doc.page_content[:300] for doc in docs])
        print("STEP 3: Context ready")
        prompt = f"""
Answer based on context:

Question:
{query}

Context:
{context}
"""
        print("STEP 4: Calling LLM...")
        response = llm.invoke(prompt[:2000])
        print("STEP 5: LLM responded")
        state.result = response.content
        print("STEP 6: Returning result")

    except Exception as e:
        state.result = f"Error: {str(e)}"

    return state