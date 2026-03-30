import time

def evaluate_response(query: str, response: str, start_time: float):
    """
    Basic evaluation metrics
    """

    latency = time.time() - start_time

    metrics = {
        "query": query,
        "response_length": len(response),
        "latency_seconds": round(latency, 3)
    }

    return metrics