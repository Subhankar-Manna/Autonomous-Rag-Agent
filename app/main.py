from app.api import app
from app.graph.orchestrator import run_graph
from app.graph.state import AgentState


def run_rag_pipeline(query):
    state = AgentState(user_query=query)

    final_state = run_graph(state)

    return final_state.result
