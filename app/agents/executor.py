from dotenv import load_dotenv
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from app.graph.state import AgentState

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "IJNRD2506195.pdf"
VECTOR_DB_PATH = BASE_DIR / "faiss_index"


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


def executor_agent(state: AgentState) -> AgentState:
    print("EXECUTOR CALLED")

    try:
        query = state.user_query

       
        if not DATA_PATH.exists():
            print(" PDF NOT FOUND → LLM MODE")

            response = llm.invoke(f"Explain clearly:\n{query}")
            state.result = response.content
            return state

   
        print(" PDF FOUND → RAG MODE")

        loader = PyPDFLoader(str(DATA_PATH))
        documents = loader.load()

        if not documents:
            print(" Empty PDF → LLM fallback")
            response = llm.invoke(query)
            state.result = response.content
            return state

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector store
        if VECTOR_DB_PATH.exists():
            vectorstore = FAISS.load_local(
                str(VECTOR_DB_PATH),
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(str(VECTOR_DB_PATH))

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(query)

   
        if not retrieved_docs:
            print(" No docs → LLM")
            response = llm.invoke(query)
            state.result = response.content
            return state

        # Build context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

       
        prompt = f"""
Answer using ONLY the context below.

Rules:
- Simple explanation
- Do not copy text
- If not found say "I don't know"

Question:
{query}

Context:
{context}

Answer:
"""
        response = llm.invoke(prompt)
        state.result = response.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        state.result = f"Executor Error: {str(e)}"

    return state