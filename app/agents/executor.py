from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from pathlib import Path

from app.graph.state import AgentState

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_DB_PATH = BASE_DIR / "rag_db"

# LLM (fast model)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

embeddings = None
vectorstore = None
retriever = None

if VECTOR_DB_PATH.exists():
    print("FAISS FOUND (will load on first query)")
else:
    print("FAISS NOT FOUND")


def load_retriever():
    global embeddings, vectorstore, retriever

    if retriever is None:
        print("LOADING EMBEDDINGS + FAISS...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.load_local(
            str(VECTOR_DB_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        print("FAISS READY")

    return retriever


def executor_agent(state: AgentState) -> AgentState:
    print("EXECUTOR STARTED")

    try:
        query = state.user_query
        print("Query:", query)

       
        docs = []
        if VECTOR_DB_PATH.exists():
            retriever_instance = load_retriever()
            docs = retriever_instance.invoke(query)
            print(f"Retrieved {len(docs)} docs")
        else:
            print("No vector DB found")

        
        context = "\n\n".join([doc.page_content[:200] for doc in docs])

        prompt = f"""
Answer clearly and concisely.

Question:
{query}

Context:
{context}
"""

        print("Calling LLM...")

      
        response = llm.invoke(prompt[:1500])

        print("LLM DONE")

        state.result = response.content.strip()

    except Exception as e:
        print("ERROR:", str(e))
        state.result = f"Error: {str(e)}"

    return state