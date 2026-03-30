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

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


def executor_agent(state: AgentState) -> AgentState:
    print("EXECUTOR CALLED")

    try:
        test = llm.invoke("What is machine learning?")
        print("TEST LLM:", test.content)

        # Load PDF
        if not DATA_PATH.exists():
            state.result = f"PDF not found at {DATA_PATH}"
            return state

        loader = PyPDFLoader(str(DATA_PATH))
        documents = loader.load()

        if not documents:
            state.result = "No documents found"
            return state

        # Split text
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

        # Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(state.user_query)


        query = state.user_query

        # No docs → LLM
        if not retrieved_docs or len(retrieved_docs) == 0:
            print("No docs → LLM")

            prompt = f"""
Explain clearly:

{query}
"""
            response = llm.invoke(prompt)
            state.result = response.content
            return state

        # Build context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Check relevance using LLM
        relevance_prompt = f"""
You are an AI system.

Check if the context is relevant to the question.

Question:
{query}

Context:
{context}

Answer only YES or NO.
"""
        relevance = llm.invoke(relevance_prompt).content.strip().lower()
        print("RELEVANCE:", relevance)

        # If NOT relevant → LLM
        if "no" in relevance:
            print("Not relevant → LLM")

            prompt = f"""
Explain clearly:

{query}
"""
            response = llm.invoke(prompt)
            state.result = response.content
            return state

        # RAG MODE
        prompt = f"""
Answer using ONLY the context below.

Rules:
- Be clear
- Do NOT copy text
- Explain in simple words
- If not found, say "I don't know"

Question:
{query}

Context:
{context}

Answer:
"""
        response = llm.invoke(prompt)
        print("RAG RESPONSE:", response.content)

        # Safety fallback
        if "i don't know" in response.content.lower():
            print("Weak context → fallback LLM")

            fallback_prompt = f"""
Explain clearly:

{query}
"""
            fallback_response = llm.invoke(fallback_prompt)
            state.result = fallback_response.content
        else:
            state.result = response.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        state.result = f"Executor Agent failed: {str(e)}"

    return state