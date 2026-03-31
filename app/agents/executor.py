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

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "IJNRD2506195.pdf"
VECTOR_DB_PATH = BASE_DIR / "faiss_index"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


def executor_agent(state: AgentState) -> AgentState:
    print("EXECUTOR CALLED")

    try:
        query = state.user_query

        # CHECK PDF FIRST 
        if not DATA_PATH.exists():
            print(f"PDF NOT FOUND → {DATA_PATH}")

            response = llm.invoke(f"Explain clearly:\n{query}")
            state.result = response.content
            return state

       
        loader = PyPDFLoader(str(DATA_PATH))
        documents = loader.load()

        if not documents:
            response = llm.invoke(f"Explain clearly:\n{query}")
            state.result = response.content
            return state

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector DB
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
            response = llm.invoke(f"Explain clearly:\n{query}")
            state.result = response.content
            return state

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        rag_prompt = f"""
Answer using ONLY the context below.

Question:
{query}

Context:
{context}
"""
        response = llm.invoke(rag_prompt)

        state.result = response.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        state.result = f"Executor failed: {str(e)}"

    return state