import json
from datetime import datetime, timezone

import numpy as np
import pytest

from src.embeddings.local import (
    encode_texts,
    generate_embeddings,
    load_retrieval_records,
    write_embedding_artifacts,
)


class FakeModel:
    def __init__(self):
        self.call = None

    def encode(self, sentences, **kwargs):
        self.call = (list(sentences), kwargs)
        vectors = np.asarray([[len(text), index + 1] for index, text in enumerate(sentences)])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def write_jsonl(path, records) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_loads_unique_nonempty_retrieval_records(tmp_path) -> None:
    path = tmp_path / "documents.jsonl"
    write_jsonl(
        path,
        [
            {"document_id": "doc-1", "text": " Sol och bad "},
            {"document_id": "doc-2", "text": "Porto atmosphere"},
        ],
    )

    document_ids, texts = load_retrieval_records(path)

    assert document_ids == ["doc-1", "doc-2"]
    assert texts == ["Sol och bad", "Porto atmosphere"]


def test_rejects_duplicate_document_ids(tmp_path) -> None:
    path = tmp_path / "documents.jsonl"
    write_jsonl(
        path,
        [
            {"document_id": "doc-1", "text": "One"},
            {"document_id": "doc-1", "text": "Two"},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate document_id"):
        load_retrieval_records(path)


def test_encodes_normalized_float32_vectors_with_expected_options() -> None:
    model = FakeModel()

    vectors = encode_texts(model, ["one", "two"], batch_size=2)

    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert model.call == (
        ["one", "two"],
        {
            "batch_size": 2,
            "show_progress_bar": True,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        },
    )


def test_rejects_nonfinite_vectors() -> None:
    class BrokenModel:
        def encode(self, sentences, **kwargs):
            return [[float("nan"), 0.0]]

    with pytest.raises(ValueError, match="NaN or infinite"):
        encode_texts(BrokenModel(), ["text"])


def test_writes_mapped_vectors_and_manifest_without_overwrite(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"document_id":"doc-1","text":"hej"}\n', encoding="utf-8")
    vectors = np.asarray([[0.6, 0.8]], dtype=np.float32)
    created_at = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)

    vectors_path, manifest_path = write_embedding_artifacts(
        ["doc-1"], vectors, source, "test-model", tmp_path, created_at
    )
    stored = np.load(vectors_path, allow_pickle=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert stored["document_ids"].tolist() == ["doc-1"]
    assert np.array_equal(stored["embeddings"], vectors)
    assert manifest["model"] == "test-model"
    assert manifest["dimensions"] == 2
    assert manifest["document_count"] == 1
    assert len(manifest["source_sha256"]) == 64
    with pytest.raises(FileExistsError):
        write_embedding_artifacts(
            ["doc-1"], vectors, source, "test-model", tmp_path, created_at
        )


def test_generate_embeddings_uses_injected_model(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [{"document_id": "doc-1", "text": "sunshine"}])

    vectors_path, manifest_path = generate_embeddings(
        source,
        tmp_path / "output",
        model_name="fake-model",
        batch_size=1,
        model=FakeModel(),
    )

    assert vectors_path.exists()
    assert manifest_path.exists()
