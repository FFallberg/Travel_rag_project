"""Create embedding-ready JSONL documents from cleaned travel threads."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path("data/processed")


def _validate_attribution(item: dict[str, Any], label: str) -> None:
    """Require the fields needed to attribute licensed source content."""
    license_name = item.get("content_license")
    author = item.get("author")
    if not isinstance(license_name, str) or not license_name.strip():
        raise ValueError(f"{label} is missing its content license")
    if not isinstance(author, dict):
        raise ValueError(f"{label} is missing author attribution")
    display_name = author.get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError(f"{label} is missing the author's display name")


def load_thread(path: Path) -> dict[str, Any]:
    """Load and validate the core shape of one cleaned thread document."""
    try:
        thread = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}: {error}") from error

    if not isinstance(thread, dict):
        raise ValueError(f"Thread {path} must contain a JSON object")
    if thread.get("source") != "Travel Stack Exchange":
        raise ValueError(f"Thread {path} is not from Travel Stack Exchange")
    question = thread.get("question")
    answers = thread.get("answers")
    if not isinstance(question, dict) or not isinstance(answers, list):
        raise ValueError(f"Thread {path} is missing question or answers data")
    question_id = question.get("question_id")
    if not isinstance(question_id, int) or question_id <= 0:
        raise ValueError(f"Thread {path} has an invalid question_id")
    if not isinstance(thread.get("source_url"), str) or not thread["source_url"]:
        raise ValueError(f"Thread {path} is missing its source_url")
    if not isinstance(question.get("title"), str) or not question["title"].strip():
        raise ValueError(f"Thread {path} is missing its question title")
    if not isinstance(question.get("text"), str) or not question["text"].strip():
        raise ValueError(f"Thread {path} is missing its question text")
    return thread


def _metadata(
    thread: dict[str, Any],
    item: dict[str, Any],
    content_type: str,
) -> dict[str, Any]:
    question = thread["question"]
    return {
        "source": thread["source"],
        "content_type": content_type,
        "question_id": question["question_id"],
        "answer_id": item.get("answer_id"),
        "question_title": question["title"],
        "tags": question.get("tags", []),
        "score": item.get("score"),
        "is_accepted": item.get("is_accepted") if content_type == "answer" else None,
        "author": item.get("author"),
        "created_at": item.get("created_at"),
        "last_activity_at": item.get("last_activity_at"),
        "collected_at": thread.get("collected_at"),
    }


def build_retrieval_documents(thread: dict[str, Any]) -> list[dict[str, Any]]:
    """Create one retrieval unit for a question and one for each answer."""
    question = thread["question"]
    question_id = question["question_id"]
    title = question["title"].strip()
    _validate_attribution(question, f"Question {question_id}")
    documents = [
        {
            "document_id": f"stackexchange-question-{question_id}",
            "text": f"Question: {title}\n\n{question['text'].strip()}",
            "source_url": thread["source_url"],
            "content_license": question.get("content_license"),
            "metadata": _metadata(thread, question, "question"),
        }
    ]

    seen_answer_ids: set[int] = set()
    for answer in thread["answers"]:
        if not isinstance(answer, dict):
            raise ValueError(f"Question {question_id} contains a non-object answer")
        answer_id = answer.get("answer_id")
        if not isinstance(answer_id, int) or answer_id <= 0:
            raise ValueError(f"Question {question_id} contains an invalid answer_id")
        if answer_id in seen_answer_ids:
            raise ValueError(f"Question {question_id} contains duplicate answer_id {answer_id}")
        seen_answer_ids.add(answer_id)
        if answer.get("question_id") != question_id:
            raise ValueError(f"Answer {answer_id} does not belong to question {question_id}")
        _validate_attribution(answer, f"Answer {answer_id}")
        answer_text = answer.get("text")
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise ValueError(f"Answer {answer_id} is missing text")
        source_url = answer.get("source_url")
        if not isinstance(source_url, str) or not source_url:
            raise ValueError(f"Answer {answer_id} is missing its source_url")
        documents.append(
            {
                "document_id": f"stackexchange-answer-{answer_id}",
                "text": f"Question: {title}\n\nAnswer: {answer_text.strip()}",
                "source_url": source_url,
                "content_license": answer.get("content_license"),
                "metadata": _metadata(thread, answer, "answer"),
            }
        )
    return documents


def load_documents(input_dir: Path) -> list[dict[str, Any]]:
    """Build deterministic retrieval documents from all thread files in a run."""
    paths = sorted(input_dir.glob("question_*.json"))
    if not paths:
        raise ValueError(f"No cleaned question files found in {input_dir}")

    documents: list[dict[str, Any]] = []
    seen_document_ids: set[str] = set()
    for path in paths:
        for document in build_retrieval_documents(load_thread(path)):
            document_id = document["document_id"]
            if document_id in seen_document_ids:
                raise ValueError(f"Duplicate retrieval document_id: {document_id}")
            seen_document_ids.add(document_id)
            documents.append(document)
    return documents


def write_jsonl(
    documents: list[dict[str, Any]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    created_at: datetime | None = None,
) -> Path:
    """Write retrieval documents as JSONL without overwriting earlier output."""
    timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"stackexchange_retrieval_{timestamp:%Y%m%dT%H%M%SZ}.jsonl"
    with path.open("x", encoding="utf-8") as output:
        for document in documents:
            output.write(json.dumps(document, ensure_ascii=False))
            output.write("\n")
    return path


def process_directory(input_dir: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Convert a cleaned Stack Exchange run into retrieval-ready JSONL."""
    return write_jsonl(load_documents(input_dir), output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        path = process_directory(args.input_dir, args.output_dir)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"Retrieval document processing failed: {error}") from error
    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
    print(f"Saved {count} retrieval documents to {path}")


if __name__ == "__main__":
    main()
