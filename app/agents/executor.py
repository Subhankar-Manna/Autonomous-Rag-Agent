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
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Load embeddings ONCE
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


if VECTOR_DB_PATH.exists():
    vectorstore = FAISS.load_local(
        str(VECTOR_DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    loader = PyPDFLoader(str(DATA_PATH))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTOR_DB_PATH))


def executor_agent(state: AgentState) -> AgentState:
    print("EXECUTOR CALLED")

    try:
        query = state.user_query

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(query)

        # No docs → LLM
        if not retrieved_docs:
            response = llm.invoke(f"Explain clearly:\n{query}")
            state.result = response.content
            return state

        # Build context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        #  Relevance check
        relevance_prompt = f"""
Check if the context is relevant to the question.

Question:
{query}

Context:
{context}

Answer only YES or NO.
"""
        relevance = llm.invoke(relevance_prompt).content.strip().lower()

        
        if "no" in relevance:
            response = llm.invoke(f"Explain clearly:\n{query}")
            state.result = response.content
            return state

       
        rag_prompt = f"""
Answer using ONLY the context below.

- Be simple
- Do not copy text
- If not found say "I don't know"

Question:
{query}

Context:
{context}

Answer:
"""
        response = llm.invoke(rag_prompt)

       
        if "i don't know" in response.content.lower():
            fallback = llm.invoke(f"Explain clearly:\n{query}")
            state.result = fallback.content
        else:
            state.result = response.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        state.result = f"Executor Agent failed: {str(e)}"

    return state