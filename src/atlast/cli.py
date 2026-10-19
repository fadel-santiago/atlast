"""Command-line entry point for local ingestion and search."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from atlast.embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder
from atlast.errors import AtlastError
from atlast.index import DEFAULT_COLLECTION, VectorIndex
from atlast.qa import DEFAULT_MAX_CONTEXT_CHARS, OllamaGenerator, QuestionAnsweringService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlast", description="ATLAST local document index")
    parser.add_argument("--db", type=Path, default=Path(".atlast/qdrant"), help="Qdrant data path")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformers model name/path")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Index one or more PDF/TXT files")
    ingest_parser.add_argument("paths", nargs="+", type=Path)
    ingest_parser.add_argument("--chunk-size", type=int, default=200, help="Words per chunk")
    ingest_parser.add_argument("--overlap", type=int, default=40, help="Overlapping words")

    search_parser = subparsers.add_parser("search", help="Run a similarity search")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)

    ask_parser = subparsers.add_parser("ask", help="Answer a question from indexed passages")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--limit", type=int, default=4, help="Retrieved passage count")
    ask_parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
        help="Installed Ollama generation model",
    )
    ask_parser.add_argument(
        "--ollama-url",
        default=os.environ.get("ATLAST_OLLAMA_URL", "http://127.0.0.1:11434"),
    )
    ask_parser.add_argument("--timeout", type=float, default=180.0)
    ask_parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Measure retrieval against the packaged labeled dataset"
    )
    evaluate_parser.add_argument("--dataset", type=Path)
    evaluate_parser.add_argument("--k", type=int, default=3)

    serve_parser = subparsers.add_parser("serve", help="Run the local ATLAST web interface")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
        help="Installed Ollama generation model",
    )
    serve_parser.add_argument(
        "--ollama-url",
        default=os.environ.get("ATLAST_OLLAMA_URL", "http://127.0.0.1:11434"),
    )
    serve_parser.add_argument("--timeout", type=float, default=180.0)
    serve_parser.add_argument("--max-upload-mb", type=int, default=10)
    serve_parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "serve":
        from atlast.web import WebConfig, run_server

        if not 1 <= args.port <= 65535:
            print("error: port must be between 1 and 65535", file=sys.stderr)
            return 2
        if args.max_upload_mb <= 0:
            print("error: max-upload-mb must be greater than zero", file=sys.stderr)
            return 2
        if args.max_context_chars <= 0:
            print("error: max-context-chars must be greater than zero", file=sys.stderr)
            return 2
        config = WebConfig(
            db_path=args.db,
            collection=args.collection,
            embedding_model=args.model,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            ollama_timeout=args.timeout,
            max_upload_bytes=args.max_upload_mb * 1024 * 1024,
            max_context_chars=args.max_context_chars,
        )
        run_server(config, port=args.port)
        return 0

    embedder = SentenceTransformerEmbedder(args.model)

    try:
        if args.command == "ingest":
            with VectorIndex(
                args.db,
                embedder,
                collection=args.collection,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            ) as index:
                results = [index.ingest(path).to_dict() for path in args.paths]
            print(json.dumps(results, indent=2, sort_keys=True))
            return 0

        if args.command == "search":
            with VectorIndex(args.db, embedder, collection=args.collection) as index:
                hits = [hit.to_dict() for hit in index.search(args.query, limit=args.limit)]
            print(json.dumps(hits, indent=2, sort_keys=True))
            return 0

        if args.command == "evaluate":
            from atlast.evaluation import DEFAULT_DATASET, run_evaluation

            report = run_evaluation(args.dataset or DEFAULT_DATASET, embedder, k=args.k)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        generator = OllamaGenerator(
            args.ollama_model,
            base_url=args.ollama_url,
            timeout=args.timeout,
        )
        with VectorIndex(args.db, embedder, collection=args.collection) as index:
            service = QuestionAnsweringService(
                index,
                generator,
                retrieval_limit=args.limit,
                max_context_chars=args.max_context_chars,
            )
            response = service.ask(args.question)
        print(json.dumps(response.to_dict(), indent=2, sort_keys=True))
        return 0 if response.status in {"answered", "unsupported", "empty_index"} else 3
    except (AtlastError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
