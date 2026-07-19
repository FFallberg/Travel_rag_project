"""Evaluate semantic retrieval against manually judged queries."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.embeddings.local import EmbeddingModel, create_local_model
from src.retrieval.semantic_search import SearchIndex, load_search_index, search


@dataclass(frozen=True)
class EvaluationCase:
    """A query with one or more manually judged relevant documents."""

    case_id: str
    query: str
    relevant_document_ids: frozenset[str]


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate a JSON evaluation set."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read evaluation cases {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Evaluation file must contain a cases list")

    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()
    for position, item in enumerate(payload["cases"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation case {position} must be a JSON object")
        case_id = item.get("case_id")
        query = item.get("query")
        relevant_ids = item.get("relevant_document_ids")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Evaluation case {position} has an invalid case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate evaluation case_id: {case_id}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Evaluation case {case_id} has an empty query")
        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or any(not isinstance(document_id, str) or not document_id for document_id in relevant_ids)
        ):
            raise ValueError(f"Evaluation case {case_id} has invalid relevant document IDs")
        if len(set(relevant_ids)) != len(relevant_ids):
            raise ValueError(f"Evaluation case {case_id} has duplicate relevant document IDs")
        seen_case_ids.add(case_id)
        cases.append(EvaluationCase(case_id, query.strip(), frozenset(relevant_ids)))

    if not cases:
        raise ValueError("Evaluation set must contain at least one case")
    return cases


def evaluate(
    index: SearchIndex,
    cases: list[EvaluationCase],
    top_k: int = 3,
    model: EmbeddingModel | None = None,
) -> dict[str, Any]:
    """Compute Hit Rate, mean reciprocal rank, and mean recall at k."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not cases:
        raise ValueError("at least one evaluation case is required")
    index_ids = set(index.document_ids.tolist())
    for case in cases:
        unknown_ids = case.relevant_document_ids - index_ids
        if unknown_ids:
            raise ValueError(
                f"Evaluation case {case.case_id} references unknown documents: "
                f"{', '.join(sorted(unknown_ids))}"
            )

    active_model = model or create_local_model(index.model_name)
    query_reports: list[dict[str, Any]] = []
    hits = 0
    reciprocal_rank_sum = 0.0
    recall_sum = 0.0
    for case in cases:
        results = search(index, case.query, top_k=top_k, model=active_model)
        retrieved_ids = [result["document_id"] for result in results]
        relevant_ranks = [
            rank
            for rank, document_id in enumerate(retrieved_ids, start=1)
            if document_id in case.relevant_document_ids
        ]
        first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
        relevant_retrieved = len(relevant_ranks)
        hit = first_relevant_rank is not None
        recall = relevant_retrieved / len(case.relevant_document_ids)
        hits += int(hit)
        reciprocal_rank_sum += 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        recall_sum += recall
        query_reports.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "relevant_document_ids": sorted(case.relevant_document_ids),
                "hit": hit,
                "first_relevant_rank": first_relevant_rank,
                "recall_at_k": recall,
                "retrieved": [
                    {
                        "rank": rank,
                        "document_id": result["document_id"],
                        "score": result["score"],
                        "is_relevant": result["document_id"] in case.relevant_document_ids,
                        "source_url": result["source_url"],
                    }
                    for rank, result in enumerate(results, start=1)
                ],
            }
        )

    query_count = len(cases)
    return {
        "summary": {
            "query_count": query_count,
            "top_k": top_k,
            "hit_rate_at_k": hits / query_count,
            "mrr_at_k": reciprocal_rank_sum / query_count,
            "mean_recall_at_k": recall_sum / query_count,
        },
        "queries": query_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--documents-file", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        index = load_search_index(args.manifest, args.documents_file)
        report = evaluate(index, load_cases(args.cases), args.top_k)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Retrieval evaluation failed: {error}") from error
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
