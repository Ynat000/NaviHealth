import json
from pathlib import Path
from datetime import datetime

LOG_PATH = Path(__file__).parent.parent / "eval_results" / "all_calls.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)


def log_call(source: str, query: str, answer: str, sources: list[str], scores: dict = None):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "query": query,
        "answer": answer,
        "sources": sources,
        "scores": scores,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
