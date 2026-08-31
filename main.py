from __future__ import annotations

import argparse
import sys

from rag import LocalRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline local RAG assistant powered by Microsoft Foundry Local."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuilding the SQLite document index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of document chunks retrieved per question (default: 3).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rag = LocalRAG(top_k=args.top_k, force_reindex=args.reindex)

    print("=" * 62)
    print(" Local RAG Simple - Microsoft Foundry Local + SQLite")
    print("=" * 62)

    try:
        rag.start()

        print("\nReady.")
        print("Ask a question about the documents in the docs/ folder.")
        print("Type q, quit, or exit to close.\n")

        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not question:
                continue
            if question.lower() in {"q", "quit", "exit"}:
                break

            answer, sources = rag.answer(question)

            print(f"\nAssistant: {answer}\n")
            if sources:
                print("Sources:")
                for item in sources:
                    print(
                        f"  - {item.source} | chunk {item.chunk_index} | "
                        f"similarity {item.score:.3f}"
                    )
                print()

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "\nCheck README.md for setup requirements. "
            r"If the database is stale/corrupt, run: .\.venv\Scripts\python.exe main.py --reindex",
            file=sys.stderr,
        )
        return 1
    finally:
        rag.close()

    print("Local models unloaded. Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
