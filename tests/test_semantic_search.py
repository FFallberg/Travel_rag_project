import hashlib
import json

import numpy as np
import pytest

from src.retrieval.semantic_search import load_search_index, search


class QueryModel:
    def __init__(self, vector):
        self.vector = np.asarray([vector], dtype=np.float32)
        self.vector /= np.linalg.norm(self.vector, axis=1, keepdims=True)

    def encode(self, sentences, **kwargs):
        return self.vector


def create_index_files(tmp_path):
    documents_path = tmp_path / "documents.jsonl"
    records = [
        {
            "document_id": "doc-water",
            "text": "Swimming in a sheltered bay",
            "source_url": "https://example.com/water",
            "content_license": "CC BY-SA 4.0",
            "metadata": {"tags": ["beaches"]},
        },
        {
            "document_id": "doc-train",
            "text": "Taking a night train",
            "source_url": "https://example.com/train",
            "content_license": "CC BY-SA 4.0",
            "metadata": {"tags": ["trains"]},
        },
    ]
    documents_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    vectors_path = tmp_path / "embeddings.npz"
    with vectors_path.open("wb") as output:
        np.savez_compressed(
            output,
            document_ids=np.asarray(["doc-water", "doc-train"]),
            embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "model": "fake-model",
        "dimensions": 2,
        "document_count": 2,
        "source_file": str(documents_path),
        "source_sha256": hashlib.sha256(documents_path.read_bytes()).hexdigest(),
        "vectors_file": vectors_path.name,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, documents_path


def test_loads_index_and_ranks_cosine_similarity(tmp_path) -> None:
    manifest_path, _ = create_index_files(tmp_path)
    index = load_search_index(manifest_path)

    results = search(index, "bad nära vatten", top_k=2, model=QueryModel([0.8, 0.6]))

    assert [result["document_id"] for result in results] == ["doc-water", "doc-train"]
    assert results[0]["score"] == pytest.approx(0.8)
    assert results[0]["source_url"] == "https://example.com/water"
    assert results[0]["metadata"] == {"tags": ["beaches"]}


def test_tied_scores_keep_artifact_order(tmp_path) -> None:
    manifest_path, _ = create_index_files(tmp_path)
    index = load_search_index(manifest_path)

    results = search(index, "equal", top_k=10, model=QueryModel([1.0, 1.0]))

    assert [result["document_id"] for result in results] == ["doc-water", "doc-train"]


def test_rejects_changed_retrieval_documents(tmp_path) -> None:
    manifest_path, documents_path = create_index_files(tmp_path)
    documents_path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_search_index(manifest_path)


def test_rejects_query_with_wrong_dimensions(tmp_path) -> None:
    manifest_path, _ = create_index_files(tmp_path)
    index = load_search_index(manifest_path)

    with pytest.raises(ValueError, match="index expects 2"):
        search(index, "query", model=QueryModel([1.0, 0.0, 0.0]))


@pytest.mark.parametrize(("query", "top_k"), [("", 1), ("valid", 0)])
def test_rejects_invalid_search_arguments(tmp_path, query, top_k) -> None:
    manifest_path, _ = create_index_files(tmp_path)
    index = load_search_index(manifest_path)

    with pytest.raises(ValueError):
        search(index, query, top_k=top_k, model=QueryModel([1.0, 0.0]))
