"""Collect raw travel questions and answers from the Stack Exchange API."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

QUESTIONS_URL = "https://api.stackexchange.com/2.3/questions"
SITE = "travel"
DEFAULT_OUTPUT_DIR = Path("data/raw")
MAX_QUESTIONS = 100
MAX_ANSWERS = 100
VALID_SORTS = ("activity", "creation", "votes", "hot", "week", "month")


def validate_limit(limit: int) -> int:
    """Return a Stack Exchange-compatible page size."""
    if not 1 <= limit <= MAX_QUESTIONS:
        raise ValueError(f"limit must be between 1 and {MAX_QUESTIONS}")
    return limit


def fetch_questions(
    limit: int,
    sort: str = "votes",
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Return the Travel Stack Exchange response without transforming it."""
    validate_limit(limit)
    if sort not in VALID_SORTS:
        raise ValueError(f"sort must be one of: {', '.join(VALID_SORTS)}")

    params: dict[str, str | int] = {
        "site": SITE,
        "pagesize": limit,
        "page": 1,
        "order": "desc",
        "sort": sort,
        "filter": "withbody",
    }
    if api_key:
        params["key"] = api_key

    http = session or requests.Session()
    response = http.get(QUESTIONS_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Stack Exchange response was not a JSON object")
    return payload


def extract_question_ids(response: dict[str, Any]) -> list[int]:
    """Extract unique positive question IDs needed for the answers endpoint."""
    question_ids: list[int] = []
    for item in response.get("items", []):
        question_id = item.get("question_id") if isinstance(item, dict) else None
        if (
            isinstance(question_id, int)
            and question_id > 0
            and question_id not in question_ids
        ):
            question_ids.append(question_id)
    return question_ids


def answers_url(question_ids: list[int]) -> str:
    """Build the batched answers endpoint for validated question IDs."""
    if not question_ids:
        raise ValueError("at least one question ID is required")
    if len(question_ids) > MAX_QUESTIONS:
        raise ValueError(f"at most {MAX_QUESTIONS} question IDs are allowed")
    if any(not isinstance(question_id, int) or question_id <= 0 for question_id in question_ids):
        raise ValueError("question IDs must be positive integers")
    joined_ids = ";".join(str(question_id) for question_id in question_ids)
    return f"{QUESTIONS_URL}/{joined_ids}/answers"


def fetch_answers(
    question_ids: list[int],
    limit: int = MAX_ANSWERS,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Return answers for a question batch without transforming the response."""
    validate_limit(limit)
    params: dict[str, str | int] = {
        "site": SITE,
        "pagesize": limit,
        "page": 1,
        "order": "desc",
        "sort": "votes",
        "filter": "withbody",
    }
    if api_key:
        params["key"] = api_key

    http = session or requests.Session()
    response = http.get(answers_url(question_ids), params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Stack Exchange answers response was not a JSON object")
    return payload


def save_raw_response(
    response: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    collected_at: datetime | None = None,
    resource: str = "questions",
    endpoint: str = QUESTIONS_URL,
) -> Path:
    """Save an unmodified API response plus separate collection metadata."""
    if resource not in {"questions", "answers"}:
        raise ValueError("resource must be 'questions' or 'answers'")
    timestamp = (collected_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"stackexchange_{resource}_{timestamp:%Y%m%dT%H%M%SZ}.json"
    document = {
        "collection": {
            "collected_at": timestamp.isoformat().replace("+00:00", "Z"),
            "source": "Travel Stack Exchange",
            "site": SITE,
            "resource": resource,
            "endpoint": endpoint,
            "license_note": (
                "Each item's content_license field governs that item; attribution "
                "and ShareAlike requirements apply."
            ),
        },
        "response": response,
    }
    with path.open("x", encoding="utf-8") as output:
        json.dump(document, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25, help="Questions to fetch (1-100)")
    parser.add_argument("--sort", choices=VALID_SORTS, default="votes")
    parser.add_argument(
        "--answers-limit",
        type=int,
        default=MAX_ANSWERS,
        help="Maximum answers across the selected questions (1-100)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    api_key = os.getenv("STACKEXCHANGE_API_KEY")
    try:
        questions_response = fetch_questions(
            limit=args.limit,
            sort=args.sort,
            api_key=api_key,
        )
        question_ids = extract_question_ids(questions_response)
        if not question_ids:
            raise RuntimeError("Question response did not contain any question IDs")
        answer_endpoint = answers_url(question_ids)
        answers_response = fetch_answers(
            question_ids,
            limit=args.answers_limit,
            api_key=api_key,
        )

        timestamp = datetime.now(timezone.utc)
        questions_path = save_raw_response(
            questions_response,
            args.output_dir,
            timestamp,
        )
        answers_path = save_raw_response(
            answers_response,
            args.output_dir,
            timestamp,
            resource="answers",
            endpoint=answer_endpoint,
        )
    except (requests.RequestException, RuntimeError, ValueError, FileExistsError) as error:
        raise SystemExit(f"Stack Exchange ingestion failed: {error}") from error

    question_count = len(questions_response.get("items", []))
    answer_count = len(answers_response.get("items", []))
    remaining = answers_response.get("quota_remaining", "unknown")
    print(f"Saved {question_count} untouched questions to {questions_path}")
    print(f"Saved {answer_count} untouched answers to {answers_path}")
    print(f"API quota remaining: {remaining}")


if __name__ == "__main__":
    main()
