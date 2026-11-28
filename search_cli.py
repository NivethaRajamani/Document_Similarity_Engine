"""CLI: quick terminal search (handy for debugging).

Usage:  python search_cli.py "The borrower shall not transfer its assets" -n 5
"""
import argparse

from src.pipeline import get_engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar legal clauses")
    parser.add_argument("query", help="Clause text to search with")
    parser.add_argument("-n", "--top_n", type=int, default=5)
    args = parser.parse_args()

    engine = get_engine()
    results = engine.search(args.query, top_n=args.top_n)
    for i, row in results.iterrows():
        print(f"\nResult {i + 1}")
        print("Clause Type     :", row["clause_type"])
        print("Similarity Score:", round(row["similarity_score"], 4))
        print("Legal Clause    :", row["clause_text"])
        print("-" * 100)
