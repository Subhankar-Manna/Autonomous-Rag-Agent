import os
from typing import List
from pathlib import Path
from pypdf import PdfReader
from app.graph.state import AgentState


#FIXED PATH
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

MAX_CHUNKS = 5


def extract_pdf_text() -> List[dict]:
    """
    Extract text from all PDFs in data/ with page numbers
    """
    chunks = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            file_path = DATA_DIR / file
            reader = PdfReader(file_path)

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    chunks.append({
                        "text": text.strip(),
                        "page": i + 1,
                        "source": file
                    })

    return chunks


def score_chunk(chunk_text: str, query_words: List[str]) -> int:
    score = 0
    text = chunk_text.lower()

    for word in query_words:
        if word in text:
            score += 1

    return score


def research_agent(state: AgentState) -> AgentState:
    """
    Research Agent (IMPROVED RAG)
    - Reads PDFs
    - Scores relevance
    - Selects top chunks
    - Cleans output
    """

    if state.research:
        return state

    query = state.user_query.lower()
    query_words = query.split()

    all_chunks = extract_pdf_text()

    scored_chunks = []
    for chunk in all_chunks:
        score = score_chunk(chunk["text"], query_words)
        if score > 0:
            scored_chunks.append((score, chunk))

 
    if not scored_chunks:
        state.research = "No relevant content found in uploaded documents."
        return state

   
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [c[1] for c in scored_chunks[:MAX_CHUNKS]]

    research_notes = []
    for c in top_chunks:
        clean_text = c["text"].replace("\n", " ").strip()

        research_notes.append(
            f"- {clean_text[:250]}... (Page {c['page']})"
        )

    state.research = "\n".join(research_notes)

    return state