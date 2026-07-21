"""Collect a small, configurable Travel Stack Exchange pilot corpus."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.ingestion.stackexchange import (
    MAX_ANSWERS,
    SEARCH_URL,
    answers_url,
    extract_question_ids,
    fetch_answers,
    fetch_targeted_questions,
    save_raw_response,
    validate_limit,
)

DEFAULT_CONFIG = Path("config/stackexchange_pilot.json")
DEFAULT_OUTPUT_DIR = Path("data/raw")


def load_pilot_config(path: Path) -> dict[str, Any]:
    """Load and validate a pilot-search configuration."""
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise ValueError("pilot config must be a JSON object")

    limit = config.get("limit_per_search", 10)
    minimum_answers = config.get("minimum_answers", 1)
    searches = config.get("searches")
    if not isinstance(limit, int):
        raise ValueError("limit_per_search must be an integer")
    validate_limit(limit)
    if not isinstance(minimum_answers, int) or minimum_answers < 0:
        raise ValueError("minimum_answers must be a non-negative integer")
    if not isinstance(searches, list) or not searches:
        raise ValueError("searches must be a non-empty list")

    names: set[str] = set()
    for search in searches:
        if not isinstance(search, dict):
            raise ValueError("each search must be a JSON object")
        name = search.get("name")
        query = search.get("query")
        tags = search.get("tags", [])
        if not isinstance(name, str) or not name or not name.replace("-", "").isalnum():
            raise ValueError("search names must contain only letters, numbers and hyphens")
        if name in names:
            raise ValueError(f"duplicate search name: {name}")
        names.add(name)
        if query is not None and not isinstance(query, str):
            raise ValueError(f"query for {name} must be a string")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"tags for {name} must be a list of strings")
        if not (query and query.strip()) and not tags:
            raise ValueError(f"search {name} requires a query or tags")
    return config


def unique_question_ids(responses: list[dict[str, Any]]) -> list[int]:
    """Return question IDs once, preserving their first-seen order."""
    unique: list[int] = []
    seen: set[int] = set()
    for response in responses:
        for question_id in extract_question_ids(response):
            if question_id not in seen:
                seen.add(question_id)
                unique.append(question_id)
    return unique


def batches(values: list[int], size: int = MAX_ANSWERS) -> list[list[int]]:
    """Split IDs into API-compatible batches."""
    if size < 1:
        raise ValueError("batch size must be positive")
    return [values[index : index + size] for index in range(0, len(values), size)]


def collect_pilot(
    config: dict[str, Any],
    output_dir: Path,
    api_key: str | None = None,
    session: requests.Session | None = None,
    collected_at: datetime | None = None,
) -> Path:
    """Run configured searches, save untouched responses and write a run manifest."""
    timestamp = (collected_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run_dir = output_dir / f"stackexchange_pilot_{timestamp:%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True, exist_ok=False)
    responses: list[dict[str, Any]] = []
    captures: list[str] = []

    for search in config["searches"]:
        response, request_params = fetch_targeted_questions(
            query=search.get("query"),
            tags=search.get("tags"),
            limit=config.get("limit_per_search", 10),
            minimum_answers=config.get("minimum_answers", 1),
            api_key=api_key,
            session=session,
        )
        search_dir = run_dir / search["name"]
        path = save_raw_response(
            response,
            search_dir,
            timestamp,
            endpoint=SEARCH_URL,
            request_params=request_params,
        )
        responses.append(response)
        captures.append(str(path.relative_to(run_dir)))

    question_ids = unique_question_ids(responses)
    if not question_ids:
        raise RuntimeError("pilot searches did not return any question IDs")

    answer_captures: list[str] = []
    for number, question_batch in enumerate(batches(question_ids), start=1):
        response = fetch_answers(
            question_batch,
            limit=MAX_ANSWERS,
            api_key=api_key,
            session=session,
        )
        path = save_raw_response(
            response,
            run_dir / f"answers-{number}",
            timestamp,
            resource="answers",
            endpoint=answers_url(question_batch),
        )
        answer_captures.append(str(path.relative_to(run_dir)))

    manifest = {
        "collected_at": timestamp.isoformat().replace("+00:00", "Z"),
        "question_captures": captures,
        "answer_captures": answer_captures,
        "unique_question_count": len(question_ids),
        "duplicate_question_count": sum(
            len(extract_question_ids(response)) for response in responses
        ) - len(question_ids),
    }
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        config = load_pilot_config(args.config)
        manifest = collect_pilot(
            config,
            args.output_dir,
            api_key=os.getenv("STACKEXCHANGE_API_KEY"),
        )
    except (OSError, json.JSONDecodeError, requests.RequestException, RuntimeError, ValueError) as error:
        raise SystemExit(f"Stack Exchange pilot ingestion failed: {error}") from error
    print(f"Saved pilot collection manifest to {manifest}")


if __name__ == "__main__":
    main()
