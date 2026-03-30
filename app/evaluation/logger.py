import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("logs.json")


def log_interaction(query, response, metrics):
    log_entry = {
        "timestamp": str(datetime.now()),
        "query": query,
        "response": response,
        "metrics": metrics
    }

    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)