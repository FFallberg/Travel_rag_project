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
    relevant_question_ids: frozenset[int] = frozenset()


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
        relevant_question_ids = item.get("relevant_question_ids", [])
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Evaluation case {position} has an invalid case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate evaluation case_id: {case_id}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Evaluation case {case_id} has an empty query")
        if not isinstance(relevant_ids, list) or any(
            not isinstance(document_id, str) or not document_id for document_id in relevant_ids
        ):
            raise ValueError(f"Evaluation case {case_id} has invalid relevant document IDs")
        if len(set(relevant_ids)) != len(relevant_ids):
            raise ValueError(f"Evaluation case {case_id} has duplicate relevant document IDs")
        if not isinstance(relevant_question_ids, list) or any(
            not isinstance(question_id, int) or isinstance(question_id, bool) or question_id <= 0
            for question_id in relevant_question_ids
        ):
            raise ValueError(f"Evaluation case {case_id} has invalid relevant question IDs")
        if len(set(relevant_question_ids)) != len(relevant_question_ids):
            raise ValueError(f"Evaluation case {case_id} has duplicate relevant question IDs")
        seen_case_ids.add(case_id)
        cases.append(
            EvaluationCase(
                case_id,
                query.strip(),
                frozenset(relevant_ids),
                frozenset(relevant_question_ids),
            )
        )

    if not cases:
        raise ValueError("Evaluation set must contain at least one case")
    return cases


def evaluate(
    index: SearchIndex,
    cases: list[EvaluationCase],
    top_k: int = 3,
    min_score: float | None = None,
    model: EmbeddingModel | None = None,
) -> dict[str, Any]:
    """Compute positive retrieval metrics and negative-query rejection at k."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not cases:
        raise ValueError("at least one evaluation case is required")
    index_ids = set(index.document_ids.tolist())
    index_question_ids = {
        record.get("metadata", {}).get("question_id")
        for record in index.records
        if isinstance(record.get("metadata"), dict)
    }
    for case in cases:
        unknown_ids = case.relevant_document_ids - index_ids
        if unknown_ids:
            raise ValueError(
                f"Evaluation case {case.case_id} references unknown documents: "
                f"{', '.join(sorted(unknown_ids))}"
            )
        unknown_question_ids = case.relevant_question_ids - index_question_ids
        if unknown_question_ids:
            raise ValueError(
                f"Evaluation case {case.case_id} references unknown question IDs: "
                f"{', '.join(str(value) for value in sorted(unknown_question_ids))}"
            )

    active_model = model or create_local_model(index.model_name)
    query_reports: list[dict[str, Any]] = []
    hits = 0
    reciprocal_rank_sum = 0.0
    recall_sum = 0.0
    positive_count = 0
    negative_count = 0
    negative_rejections = 0
    for case in cases:
        results = search(index, case.query, top_k=top_k, model=active_model)
        passing_results = [
            result
            for result in results
            if min_score is None or result["score"] >= min_score
        ]
        def relevance_target(result: dict[str, Any]) -> tuple[str, str | int] | None:
            if result["document_id"] in case.relevant_document_ids:
                return ("document", result["document_id"])
            metadata = result.get("metadata")
            question_id = metadata.get("question_id") if isinstance(metadata, dict) else None
            if question_id in case.relevant_question_ids:
                return ("question", question_id)
            return None

        relevant_targets = [relevance_target(result) for result in results]
        relevant_ranks = [
            rank
            for rank, target in enumerate(relevant_targets, start=1)
            if target is not None
            and (min_score is None or results[rank - 1]["score"] >= min_score)
        ]
        first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
        retrieved_targets = {
            relevant_targets[rank - 1]
            for rank in relevant_ranks
            if relevant_targets[rank - 1] is not None
        }
        relevant_target_count = len(case.relevant_document_ids) + len(case.relevant_question_ids)
        relevant_retrieved = len(retrieved_targets)
        is_positive = relevant_target_count > 0
        hit = first_relevant_rank is not None if is_positive else None
        recall = relevant_retrieved / relevant_target_count if is_positive else None
        rejected = not passing_results
        if is_positive:
            positive_count += 1
            hits += int(bool(hit))
            reciprocal_rank_sum += 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            recall_sum += float(recall)
        else:
            negative_count += 1
            negative_rejections += int(rejected)
        query_reports.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "relevant_document_ids": sorted(case.relevant_document_ids),
                "relevant_question_ids": sorted(case.relevant_question_ids),
                "hit": hit,
                "is_negative": not is_positive,
                "rejected": rejected,
                "first_relevant_rank": first_relevant_rank,
                "recall_at_k": recall,
                "retrieved": [
                    {
                        "rank": rank,
                        "document_id": result["document_id"],
                        "score": result["score"],
                        "passes_min_score": min_score is None or result["score"] >= min_score,
                        "is_relevant": relevance_target(result) is not None,
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
            "positive_query_count": positive_count,
            "negative_query_count": negative_count,
            "top_k": top_k,
            "min_score": min_score,
            "hit_rate_at_k": hits / positive_count if positive_count else None,
            "mrr_at_k": reciprocal_rank_sum / positive_count if positive_count else None,
            "mean_recall_at_k": recall_sum / positive_count if positive_count else None,
            "negative_rejection_rate": (
                negative_rejections / negative_count if negative_count else None
            ),
        },
        "queries": query_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--documents-file", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        index = load_search_index(args.manifest, args.documents_file)
        report = evaluate(
            index,
            load_cases(args.cases),
            top_k=args.top_k,
            min_score=args.min_score,
        )
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Retrieval evaluation failed: {error}") from error
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
