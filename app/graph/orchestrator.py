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

    # Entry point
    graph.set_entry_point("planner")

    # Flow
    graph.add_edge("planner", "research")
    graph.add_edge("research", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_edge("reviewer", "verifier")
    graph.add_edge("verifier", END)

    return graph.compile()


def run_graph(state: AgentState) -> AgentState:
    graph = build_graph()

    try:
        final_state = graph.invoke(state)

        if not isinstance(final_state, AgentState):
            final_state = AgentState(**final_state)

        return final_state

    except Exception as e:
        
        state.result = f"Graph execution failed: {str(e)}"
        return state