"""
Utility for sampling queries from test_queries.json.
Each eval script calls sample_queries() instead of using hardcoded lists.
"""
import json
import random
from pathlib import Path

_QUERY_BANK_PATH = Path(__file__).parent / "test_queries.json"


def _load_bank():
    with open(_QUERY_BANK_PATH, encoding="utf-8") as f:
        return json.load(f)["queries"]


def sample_queries(
    n: int,
    categories: list[str] | None = None,
    languages: list[str] | None = None,
    expected: list[str] | None = None,
    seed: int | None = None,
) -> list[tuple[str, str]]:
    """
    Return a list of (label, query) tuples sampled from the master query bank.

    Args:
        n:          Number of queries to return.
        categories: Filter by category (e.g. ["emergency", "infection"]).
                    None = all categories.
        languages:  Filter by language code (e.g. ["en", "zh"]).
                    None = all languages.
        expected:   Filter by expected outcome ("911", "811", "answer").
                    None = all outcomes.
        seed:       Random seed for reproducibility. None = random each run.

    Returns:
        List of (label, query) tuples, same format used by eval scripts.
    """
    pool = _load_bank()

    if categories:
        pool = [q for q in pool if q["category"] in categories]
    if languages:
        pool = [q for q in pool if q["language"] in languages]
    if expected:
        pool = [q for q in pool if q["expected"] in expected]

    if not pool:
        raise ValueError(
            f"No queries match filters: categories={categories}, "
            f"languages={languages}, expected={expected}"
        )

    rng = random.Random(seed)
    if n >= len(pool):
        selected = pool[:]
        rng.shuffle(selected)
    else:
        selected = rng.sample(pool, n)

    return [(q["label"], q["query"]) for q in selected]


def sample_queries_full(
    n: int,
    categories: list[str] | None = None,
    languages: list[str] | None = None,
    expected: list[str] | None = None,
    seed: int | None = None,
) -> list[dict]:
    """
    Same as sample_queries() but returns the full query dicts
    (id, label, query, language, category, expected).
    Useful when scripts need language or expected fields.
    """
    pool = _load_bank()

    if categories:
        pool = [q for q in pool if q["category"] in categories]
    if languages:
        pool = [q for q in pool if q["language"] in languages]
    if expected:
        pool = [q for q in pool if q["expected"] in expected]

    if not pool:
        raise ValueError(
            f"No queries match filters: categories={categories}, "
            f"languages={languages}, expected={expected}"
        )

    rng = random.Random(seed)
    if n >= len(pool):
        selected = pool[:]
        rng.shuffle(selected)
    else:
        selected = rng.sample(pool, n)

    return selected
