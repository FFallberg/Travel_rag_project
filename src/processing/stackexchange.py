"""Build cleaned thread documents from raw Stack Exchange API captures."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path("data/cleaned")
BLOCK_TAGS = {
    "blockquote",
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ol",
    "p",
    "pre",
    "table",
    "tr",
    "ul",
}


class _TextExtractor(HTMLParser):
    """Extract readable text while retaining meaningful block boundaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in "".join(self.parts).splitlines()]
        return "\n\n".join(line for line in lines if line)


def html_to_text(value: str) -> str:
    """Convert an HTML fragment to normalized plain text."""
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text()


def utc_iso(epoch: Any) -> str | None:
    """Convert an epoch value to an ISO UTC timestamp when present."""
    if not isinstance(epoch, (int, float)) or isinstance(epoch, bool):
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def author_from(owner: Any) -> dict[str, Any]:
    """Select attribution fields without inferring data for deleted users."""
    if not isinstance(owner, dict):
        return {"user_id": None, "display_name": "[deleted]", "profile_url": None}
    return {
        "user_id": owner.get("user_id"),
        "display_name": owner.get("display_name", "[deleted]"),
        "profile_url": owner.get("link"),
    }


def load_capture(path: Path, expected_resource: str) -> dict[str, Any]:
    """Load and validate a raw Stack Exchange capture wrapper."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}: {error}") from error

    if not isinstance(document, dict):
        raise ValueError(f"Capture {path} must contain a JSON object")
    collection = document.get("collection")
    response = document.get("response")
    if not isinstance(collection, dict) or not isinstance(response, dict):
        raise ValueError(f"Capture {path} is missing collection or response objects")
    if collection.get("source") != "Travel Stack Exchange":
        raise ValueError(f"Capture {path} is not from Travel Stack Exchange")
    if collection.get("resource") != expected_resource:
        raise ValueError(f"Capture {path} is not a {expected_resource} capture")
    if not isinstance(response.get("items"), list):
        raise ValueError(f"Capture {path} response is missing an items list")
    return document


def build_thread_documents(
    questions_capture: dict[str, Any],
    answers_capture: dict[str, Any],
) -> list[dict[str, Any]]:
    """Join raw questions and answers into cleaned, attributed thread documents."""
    questions_collection = questions_capture["collection"]
    answers_collection = answers_capture["collection"]
    collected_at = questions_collection.get("collected_at")
    if not collected_at or collected_at != answers_collection.get("collected_at"):
        raise ValueError("Question and answer captures must have matching collected_at values")

    questions = questions_capture["response"]["items"]
    answers = answers_capture["response"]["items"]
    if any(not isinstance(item, dict) for item in questions):
        raise ValueError("Every question must be a JSON object")
    question_ids = [item.get("question_id") for item in questions]
    if any(not isinstance(question_id, int) or question_id <= 0 for question_id in question_ids):
        raise ValueError("Every question must have a positive integer question_id")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Question capture contains duplicate question IDs")

    answers_by_question: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("Every answer must be a JSON object")
        answer_id = answer.get("answer_id")
        if not isinstance(answer_id, int) or answer_id <= 0:
            raise ValueError("Every answer must have a positive integer answer_id")
        question_id = answer.get("question_id")
        if question_id not in question_ids:
            raise ValueError(f"Answer references unknown question_id: {question_id}")
        answers_by_question[question_id].append(answer)

    documents: list[dict[str, Any]] = []
    for question in questions:
        question_id = question["question_id"]
        source_url = question.get("link")
        if not isinstance(source_url, str) or not source_url:
            raise ValueError(f"Question {question_id} is missing its source link")
        documents.append(
            {
                "document_id": f"stackexchange-thread-{question_id}",
                "source": "Travel Stack Exchange",
                "source_url": source_url,
                "collected_at": collected_at,
                "question": {
                    "question_id": question_id,
                    "title": html_to_text(str(question.get("title", ""))),
                    "text": html_to_text(str(question.get("body", ""))),
                    "author": author_from(question.get("owner")),
                    "score": question.get("score"),
                    "tags": question.get("tags", []),
                    "created_at": utc_iso(question.get("creation_date")),
                    "last_activity_at": utc_iso(question.get("last_activity_date")),
                    "content_license": question.get("content_license"),
                },
                "answers": [
                    {
                        "answer_id": answer.get("answer_id"),
                        "question_id": question_id,
                        "source_url": f"https://travel.stackexchange.com/a/{answer.get('answer_id')}",
                        "text": html_to_text(str(answer.get("body", ""))),
                        "author": author_from(answer.get("owner")),
                        "score": answer.get("score"),
                        "is_accepted": bool(answer.get("is_accepted", False)),
                        "created_at": utc_iso(answer.get("creation_date")),
                        "last_activity_at": utc_iso(answer.get("last_activity_date")),
                        "content_license": answer.get("content_license"),
                    }
                    for answer in answers_by_question[question_id]
                ],
            }
        )
    return documents


def write_documents(
    documents: list[dict[str, Any]],
    collected_at: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write one cleaned JSON file per thread without overwriting prior output."""
    try:
        timestamp = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid collected_at timestamp: {collected_at}") from error
    run_dir = output_dir / f"stackexchange_threads_{timestamp.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for document in documents:
        path = run_dir / f"question_{document['question']['question_id']}.json"
        with path.open("x", encoding="utf-8") as output:
            json.dump(document, output, ensure_ascii=False, indent=2)
            output.write("\n")
    return run_dir


def process_files(questions_file: Path, answers_file: Path, output_dir: Path) -> Path:
    """Validate, clean, join, and persist one raw capture pair."""
    questions_capture = load_capture(questions_file, "questions")
    answers_capture = load_capture(answers_file, "answers")
    documents = build_thread_documents(questions_capture, answers_capture)
    return write_documents(
        documents,
        questions_capture["collection"]["collected_at"],
        output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-file", type=Path, required=True)
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_dir = process_files(args.questions_file, args.answers_file, args.output_dir)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"Stack Exchange processing failed: {error}") from error
    count = len(list(run_dir.glob("question_*.json")))
    print(f"Saved {count} cleaned Stack Exchange threads to {run_dir}")


if __name__ == "__main__":
    main()
