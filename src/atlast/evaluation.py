"""Small reproducible retrieval evaluation for labeled local passages."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from atlast.embeddings import Embedder
from atlast.errors import EvaluationError
from atlast.index import SearchHit, VectorIndex


DEFAULT_DATASET = Path(__file__).with_name("eval_data") / "retrieval_v1.json"


def run_evaluation(
    dataset_path: str | Path,
    embedder: Embedder,
    *,
    k: int = 3,
) -> dict[str, Any]:
    """Index the fixture corpus temporarily and report Recall@K and MRR."""

    if k <= 0:
        raise ValueError("k must be greater than zero")
    path = Path(dataset_path).resolve()
    dataset = _load_dataset(path)
    document_paths = _resolve_documents(path.parent, dataset["documents"])

    case_results: list[dict[str, Any]] = []
    answerable_recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    unanswerable_with_candidates = 0

    with TemporaryDirectory(prefix="atlast-evaluation-") as temp_dir:
        with VectorIndex(
            Path(temp_dir) / "qdrant",
            embedder,
            collection="evaluation",
            chunk_size=200,
            overlap=0,
        ) as index:
            for document_path in document_paths:
                index.ingest(document_path)

            for case in dataset["cases"]:
                hits = index.search(case["question"], limit=k)
                expected = case["relevant_passages"]
                retrieved = [
                    _retrieved_passage(hit, rank, expected)
                    for rank, hit in enumerate(hits, start=1)
                ]

                result: dict[str, Any] = {
                    "id": case["id"],
                    "question": case["question"],
                    "answerable": case["answerable"],
                    "retrieved": retrieved,
                }
                if case["answerable"]:
                    found_labels = sum(
                        any(_label_matches(label, hit.metadata) for hit in hits)
                        for label in expected
                    )
                    recall = found_labels / len(expected)
                    first_relevant_rank = next(
                        (
                            rank
                            for rank, hit in enumerate(hits, start=1)
                            if any(_label_matches(label, hit.metadata) for label in expected)
                        ),
                        None,
                    )
                    reciprocal_rank = (
                        1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
                    )
                    answerable_recalls.append(recall)
                    reciprocal_ranks.append(reciprocal_rank)
                    result["recall_at_k"] = recall
                    result["reciprocal_rank"] = reciprocal_rank
                else:
                    if hits:
                        unanswerable_with_candidates += 1
                case_results.append(result)

    answerable_count = len(answerable_recalls)
    unanswerable_count = len(case_results) - answerable_count
    return {
        "dataset": dataset["name"],
        "embedding_model": embedder.name,
        "k": k,
        "documents": len(document_paths),
        "answerable_cases": answerable_count,
        "unanswerable_cases": unanswerable_count,
        "recall_at_k": _mean(answerable_recalls),
        "mrr": _mean(reciprocal_ranks),
        "unanswerable_queries_with_candidates": unanswerable_with_candidates,
        "cases": case_results,
    }


def _load_dataset(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read evaluation dataset: {path}") from exc

    if not isinstance(payload, dict):
        raise EvaluationError("Evaluation dataset must be a JSON object")
    name = payload.get("name")
    documents = payload.get("documents")
    cases = payload.get("cases")
    if not isinstance(name, str) or not name.strip():
        raise EvaluationError("Evaluation dataset requires a name")
    if not isinstance(documents, list) or not documents or not all(
        isinstance(document, str) and document for document in documents
    ):
        raise EvaluationError("Evaluation dataset requires document paths")
    if not isinstance(cases, list) or not cases:
        raise EvaluationError("Evaluation dataset requires cases")

    normalized_cases = [_validate_case(case) for case in cases]
    return {"name": name, "documents": documents, "cases": normalized_cases}


def _validate_case(case: Any) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise EvaluationError("Each evaluation case must be an object")
    case_id = case.get("id")
    question = case.get("question")
    answerable = case.get("answerable")
    relevant = case.get("relevant_passages")
    if not isinstance(case_id, str) or not case_id.strip():
        raise EvaluationError("Each evaluation case requires an id")
    if not isinstance(question, str) or not question.strip():
        raise EvaluationError(f"Evaluation case '{case_id}' requires a question")
    if not isinstance(answerable, bool):
        raise EvaluationError(f"Evaluation case '{case_id}' requires answerable=true/false")
    if not isinstance(relevant, list) or not all(isinstance(item, dict) for item in relevant):
        raise EvaluationError(f"Evaluation case '{case_id}' has invalid relevant passages")
    if answerable and not relevant:
        raise EvaluationError(f"Answerable case '{case_id}' requires a relevant passage")
    if not answerable and relevant:
        raise EvaluationError(f"Unanswerable case '{case_id}' cannot label a relevant passage")
    for label in relevant:
        if not isinstance(label.get("filename"), str):
            raise EvaluationError(f"Case '{case_id}' passage labels require filename")
    return {
        "id": case_id,
        "question": question,
        "answerable": answerable,
        "relevant_passages": relevant,
    }


def _resolve_documents(base_dir: Path, documents: list[str]) -> list[Path]:
    resolved_base = base_dir.resolve()
    resolved: list[Path] = []
    for document in documents:
        path = (resolved_base / document).resolve()
        if not path.is_relative_to(resolved_base) or not path.is_file():
            raise EvaluationError(f"Evaluation document is missing or outside the dataset: {document}")
        resolved.append(path)
    return resolved


def _retrieved_passage(
    hit: SearchHit,
    rank: int,
    expected: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = hit.metadata
    return {
        "rank": rank,
        "filename": metadata.get("filename"),
        "page": metadata.get("page"),
        "section": metadata.get("section"),
        "chunk_index": metadata.get("chunk_index"),
        "score": hit.score,
        "relevant": any(_label_matches(label, metadata) for label in expected),
    }


def _label_matches(label: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return all(metadata.get(field) == value for field, value in label.items())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
