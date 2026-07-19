import json
from datetime import datetime, timezone

import pytest

from src.processing.retrieval_documents import (
    build_retrieval_documents,
    load_documents,
    write_jsonl,
)


def thread(question_id=42, answer_id=7):
    return {
        "document_id": f"stackexchange-thread-{question_id}",
        "source": "Travel Stack Exchange",
        "source_url": f"https://travel.stackexchange.com/q/{question_id}",
        "collected_at": "2026-07-18T12:00:00Z",
        "question": {
            "question_id": question_id,
            "title": "Swimming near Porto",
            "text": "Where can I swim?",
            "author": {"user_id": 1, "display_name": "Ada", "profile_url": None},
            "score": 5,
            "tags": ["portugal", "beaches"],
            "created_at": "2026-07-17T12:00:00Z",
            "last_activity_at": "2026-07-18T12:00:00Z",
            "content_license": "CC BY-SA 4.0",
        },
        "answers": [
            {
                "answer_id": answer_id,
                "question_id": question_id,
                "source_url": f"https://travel.stackexchange.com/a/{answer_id}",
                "text": "Try the sheltered bay.",
                "author": {"user_id": 2, "display_name": "Lin", "profile_url": None},
                "score": 9,
                "is_accepted": True,
                "created_at": "2026-07-18T12:00:00Z",
                "last_activity_at": "2026-07-18T12:00:00Z",
                "content_license": "CC BY-SA 4.0",
            }
        ],
    }


def test_builds_question_and_contextual_answer_documents() -> None:
    documents = build_retrieval_documents(thread())

    assert [document["document_id"] for document in documents] == [
        "stackexchange-question-42",
        "stackexchange-answer-7",
    ]
    assert documents[0]["text"] == "Question: Swimming near Porto\n\nWhere can I swim?"
    assert documents[1]["text"] == (
        "Question: Swimming near Porto\n\nAnswer: Try the sheltered bay."
    )
    assert documents[1]["source_url"] == "https://travel.stackexchange.com/a/7"
    assert documents[1]["content_license"] == "CC BY-SA 4.0"
    assert documents[1]["metadata"]["question_id"] == 42
    assert documents[1]["metadata"]["answer_id"] == 7
    assert documents[1]["metadata"]["is_accepted"] is True
    assert documents[1]["metadata"]["author"]["display_name"] == "Lin"


def test_rejects_duplicate_answer_ids() -> None:
    duplicate = thread()
    duplicate["answers"].append(dict(duplicate["answers"][0]))
    with pytest.raises(ValueError, match="duplicate answer_id"):
        build_retrieval_documents(duplicate)


def test_rejects_answer_linked_to_another_question() -> None:
    mismatched = thread()
    mismatched["answers"][0]["question_id"] = 99
    with pytest.raises(ValueError, match="does not belong"):
        build_retrieval_documents(mismatched)


def test_requires_license_for_attribution() -> None:
    unlicensed = thread()
    unlicensed["answers"][0]["content_license"] = None
    with pytest.raises(ValueError, match="missing its content license"):
        build_retrieval_documents(unlicensed)


def test_load_documents_is_deterministic(tmp_path) -> None:
    (tmp_path / "question_99.json").write_text(json.dumps(thread(99, 8)), encoding="utf-8")
    (tmp_path / "question_42.json").write_text(json.dumps(thread(42, 7)), encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document["document_id"] for document in documents] == [
        "stackexchange-question-42",
        "stackexchange-answer-7",
        "stackexchange-question-99",
        "stackexchange-answer-8",
    ]


def test_load_documents_rejects_empty_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="No cleaned question files"):
        load_documents(tmp_path)


def test_write_jsonl_preserves_unicode_and_will_not_overwrite(tmp_path) -> None:
    documents = build_retrieval_documents(thread())
    created_at = datetime(2026, 7, 19, 10, 30, tzinfo=timezone.utc)

    path = write_jsonl(documents, tmp_path, created_at)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert lines == documents
    assert path.name == "stackexchange_retrieval_20260719T103000Z.jsonl"
    with pytest.raises(FileExistsError):
        write_jsonl(documents, tmp_path, created_at)
