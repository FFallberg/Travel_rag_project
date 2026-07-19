import json

import numpy as np
import pytest

from src.evaluation.retrieval import EvaluationCase, evaluate, load_cases
from src.retrieval.semantic_search import SearchIndex


class QueryModel:
    def __init__(self, vectors_by_query):
        self.vectors_by_query = vectors_by_query

    def encode(self, sentences, **kwargs):
        vectors = np.asarray([self.vectors_by_query[text] for text in sentences], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def index() -> SearchIndex:
    records = tuple(
        {
            "document_id": document_id,
            "text": document_id,
            "source_url": f"https://example.com/{document_id}",
            "content_license": "CC BY-SA 4.0",
            "metadata": {},
        }
        for document_id in ("doc-a", "doc-b", "doc-c")
    )
    return SearchIndex(
        model_name="fake-model",
        document_ids=np.asarray(["doc-a", "doc-b", "doc-c"]),
        embeddings=np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32),
        records=records,
    )


def test_load_cases_validates_and_deduplicates_ids(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "water", "query": "bad", "relevant_document_ids": ["doc-a"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert cases == [EvaluationCase("water", "bad", frozenset({"doc-a"}))]

    path.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "water", "query": "bad", "relevant_document_ids": ["doc-a"]},
                    {"case_id": "water", "query": "sea", "relevant_document_ids": ["doc-b"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate evaluation case_id"):
        load_cases(path)


def test_evaluate_computes_hit_mrr_and_recall() -> None:
    cases = [
        EvaluationCase("first", "query-a", frozenset({"doc-a"})),
        EvaluationCase("second", "query-c", frozenset({"doc-a", "doc-c"})),
    ]
    model = QueryModel({"query-a": [1.0, 0.0], "query-c": [0.0, 1.0]})

    report = evaluate(index(), cases, top_k=2, model=model)

    assert report["summary"] == {
        "query_count": 2,
        "top_k": 2,
        "hit_rate_at_k": 1.0,
        "mrr_at_k": 1.0,
        "mean_recall_at_k": 0.75,
    }
    assert report["queries"][0]["first_relevant_rank"] == 1
    assert report["queries"][1]["recall_at_k"] == 0.5
    assert report["queries"][0]["retrieved"][0]["is_relevant"] is True


def test_evaluate_reports_miss_and_reciprocal_rank() -> None:
    cases = [EvaluationCase("case", "query", frozenset({"doc-c"}))]
    model = QueryModel({"query": [1.0, 0.0]})

    miss = evaluate(index(), cases, top_k=1, model=model)
    found = evaluate(index(), cases, top_k=3, model=model)

    assert miss["summary"]["hit_rate_at_k"] == 0.0
    assert miss["summary"]["mrr_at_k"] == 0.0
    assert found["summary"]["mrr_at_k"] == pytest.approx(1 / 3)


def test_evaluate_rejects_unknown_relevance_id() -> None:
    cases = [EvaluationCase("case", "query", frozenset({"missing"}))]
    with pytest.raises(ValueError, match="unknown documents"):
        evaluate(index(), cases, model=QueryModel({"query": [1.0, 0.0]}))
