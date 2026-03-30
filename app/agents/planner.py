from app.graph.state import AgentState
import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"


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

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )

    response.raise_for_status()

    plan_text = response.json().get("response", "")

    # ✅ Convert plan text into clean list
    plan_steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if re.match(r"^\d+\.\s+[A-Z]", line):
           plan_steps.append(line)

    state.plan = plan_steps
    return state

