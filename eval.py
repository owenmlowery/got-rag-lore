"""
Retrieval evaluation for the GoT lore RAG index.

Runs each question in test_set.json through three retrieval strategies
(hybrid, BM25-only, KNN-only) against the `got_lore` Elasticsearch index
and reports Recall@10 and MRR per strategy.

Caveats:
- 20 hand-labeled questions is not a real benchmark. Treat the numbers as
  directional, not authoritative.
- Relevance is judged by case-insensitive substring match against a list
  of `relevant_phrases` per question, rather than hand-labeled sentence
  ids. This is easier to maintain across re-ingests but less precise
  (a sentence that mentions the topic in passing can be marked relevant).

Run:
    python eval.py
"""

import json
import os
import sys
import urllib3
from pathlib import Path

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ES_HOST = os.getenv("ES_HOST", "https://localhost:9200/")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.environ["ES_PASSWORD"]
INDEX = os.getenv("ES_INDEX", "got_lore")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
K = 10
TEST_SET_PATH = Path(__file__).parent / "test_set.json"


def retrieve_hybrid(client, model, query, k=K):
    qv = model.encode(query)
    body = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"sentence": query}},
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": qv,
                            "k": k,
                            "num_candidates": 5 * k,
                        }
                    },
                ]
            }
        },
        "size": k,
    }
    res = client.search(index=INDEX, body=body)
    return [h["_source"]["sentence"] for h in res["hits"]["hits"]]


def retrieve_bm25(client, model, query, k=K):
    body = {"query": {"match": {"sentence": query}}, "size": k}
    res = client.search(index=INDEX, body=body)
    return [h["_source"]["sentence"] for h in res["hits"]["hits"]]


def retrieve_knn(client, model, query, k=K):
    qv = model.encode(query)
    body = {
        "query": {
            "knn": {
                "field": "embedding",
                "query_vector": qv,
                "num_candidates": 5 * k,
            }
        },
        "size": k,
    }
    res = client.search(index=INDEX, body=body)
    return [h["_source"]["sentence"] for h in res["hits"]["hits"]]


def is_relevant(sentence, phrases):
    s = sentence.lower()
    return any(p.lower() in s for p in phrases)


def recall_at_k(results, phrases):
    return 1.0 if any(is_relevant(s, phrases) for s in results) else 0.0


def reciprocal_rank(results, phrases):
    for i, s in enumerate(results, start=1):
        if is_relevant(s, phrases):
            return 1.0 / i
    return 0.0


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    test_set = json.loads(TEST_SET_PATH.read_text())

    client = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False,
    )
    if not client.ping():
        print("Could not reach Elasticsearch. Check ES_HOST / credentials.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    strategies = {
        "hybrid": retrieve_hybrid,
        "bm25-only": retrieve_bm25,
        "knn-only": retrieve_knn,
    }

    scores = {name: {"recall": [], "mrr": []} for name in strategies}
    misses = {name: [] for name in strategies}

    print(f"Running {len(test_set)} questions through {len(strategies)} strategies...\n")

    for q in test_set:
        question = q["question"]
        phrases = q["relevant_phrases"]
        for name, fn in strategies.items():
            results = fn(client, model, question)
            r = recall_at_k(results, phrases)
            m = reciprocal_rank(results, phrases)
            scores[name]["recall"].append(r)
            scores[name]["mrr"].append(m)
            if r == 0.0:
                misses[name].append(question)

    n = len(test_set)
    print(f"{'Strategy':<12} {'Recall@10':>10} {'MRR':>8}")
    print("-" * 32)
    for name in strategies:
        r = sum(scores[name]["recall"]) / n
        m = sum(scores[name]["mrr"]) / n
        print(f"{name:<12} {r:>10.2f} {m:>8.2f}")
    print()
    print(f"n = {n} questions, k = {K}")

    if verbose:
        print("\nMisses (recall=0):")
        for name, qs in misses.items():
            if qs:
                print(f"  {name}:")
                for q in qs:
                    print(f"    - {q}")


if __name__ == "__main__":
    main()
