from langgraph.graph import StateGraph, END

from app.graph.state import AgentState
from app.agents.planner import planner_agent
from app.agents.research import research_agent
from app.agents.executor import executor_agent
from app.agents.reviewer import ReviewerAgent
from app.agents.verifier import VerifierAgent


def build_graph():
    graph = StateGraph(AgentState)

    reviewer = ReviewerAgent()
    verifier = VerifierAgent()

    # Add nodes 
    graph.add_node("planner", planner_agent)
    graph.add_node("research", research_agent)
    graph.add_node("executor", executor_agent)
    graph.add_node("reviewer", reviewer.run)
    graph.add_node("verifier", verifier.run)
 
    graph.set_entry_point("planner")

    # Flow 
    graph.add_edge("planner", "research")
    graph.add_edge("research", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_edge("reviewer", "verifier")
    graph.add_edge("verifier", END)

    return graph.compile()


def run_graph(user_query: str):
    graph = build_graph()

    state = {
        "user_query": user_query
    }

    result = graph.invoke(state)

    final = result.get("result", "No response generated")

    final = final.replace("\n", " ").strip()

    return final



