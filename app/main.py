from app.api import app
from app.graph.orchestrator import run_graph  

def run_rag_pipeline(query):
    result = run_graph(query)
    return result
