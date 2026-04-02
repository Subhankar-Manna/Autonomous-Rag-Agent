from fastapi import FastAPI
from pydantic import BaseModel
import time

from app.graph.orchestrator import run_graph
from app.evaluation.evaluator import evaluate_response
from app.evaluation.logger import log_interaction  

app = FastAPI(title="Autonomous Multi-Agent AI")


class QueryRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        start_time = time.time()

        # Run LangGraph pipeline
        result = run_graph(request.query)

      
        if isinstance(result, dict):
            final_answer = result.get("result", "")
        else:
            final_answer = str(result)

      
        metrics = evaluate_response(
            request.query,
            final_answer,
            start_time
        )

      
        try:
            log_interaction(request.query, final_answer, metrics)
        except Exception as log_error:
            print("Logging failed:", log_error)

    
        return {
            "response": final_answer,
            "metrics": metrics
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def home():
    return {
        "message": "AI Agent Running ",
        "status": "success"
    }