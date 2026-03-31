from app.graph.state import AgentState
import re
import os
from langchain_groq import ChatGroq


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def planner_agent(state: AgentState) -> AgentState:
    """
    Takes user_query and returns a step-by-step plan
    """

    if state.plan:
        return state

    prompt = f"""
You are a Planner Agent in a GenAI system.

RULES:
- You MUST always return a plan
- NEVER refuse
- NEVER say "I can't"
- Even if unsure, make reasonable assumptions

User query:
{state.user_query}

Return a numbered step-by-step execution plan only.
"""

    try:
        response = llm.invoke(prompt)
        plan_text = response.content
    except Exception:
        
        plan_text = "1. Understand the query\n2. Generate response\n3. Return answer"

    # Convert to list
    plan_steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if re.match(r"^\d+\.\s+", line):
            plan_steps.append(line)

    state.plan = plan_steps
    return state