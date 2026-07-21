import json

import pytest

from src.processing.stackexchange import (
    build_thread_documents,
    html_to_text,
    load_pilot_captures,
    process_files,
    process_manifest,
)


def capture(resource, items, collected_at="2026-07-18T12:00:00Z"):
    return {
        "collection": {
            "collected_at": collected_at,
            "source": "Travel Stack Exchange",
            "resource": resource,
        },
        "response": {"items": items},
    }


def test_html_to_text_preserves_blocks_and_decodes_entities() -> None:
    value = "<p>Sun &amp; sea.</p><blockquote>Bring water.<br>Stay cool.</blockquote>"
    assert html_to_text(value) == "Sun & sea.\n\nBring water.\n\nStay cool."


def test_builds_attributed_thread_and_connects_answers() -> None:
    question = {
        "question_id": 42,
        "title": "Swimming &amp; cafés",
        "body": "<p>Where can I swim?</p>",
        "link": "https://travel.stackexchange.com/q/42",
        "owner": {"user_id": 1, "display_name": "Ada", "link": "https://example.com/u/1"},
        "score": 5,
        "tags": ["beaches"],
        "creation_date": 1_721_278_800,
        "last_activity_date": 1_721_282_400,
        "content_license": "CC BY-SA 4.0",
    }
    answer = {
        "answer_id": 7,
        "question_id": 42,
        "body": "<p>Try the sheltered bay.</p>",
        "owner": {"display_name": "Lin"},
        "score": 9,
        "is_accepted": True,
        "creation_date": 1_721_282_400,
        "last_activity_date": 1_721_282_400,
        "content_license": "CC BY-SA 4.0",
    }

    documents = build_thread_documents(
        capture("questions", [question]),
        capture("answers", [answer]),
    )

    assert len(documents) == 1
    document = documents[0]
    assert document["document_id"] == "stackexchange-thread-42"
    assert document["question"]["title"] == "Swimming & cafés"
    assert document["question"]["text"] == "Where can I swim?"
    assert document["question"]["author"]["display_name"] == "Ada"
    assert document["answers"][0]["text"] == "Try the sheltered bay."
    assert document["answers"][0]["source_url"] == "https://travel.stackexchange.com/a/7"
    assert document["answers"][0]["content_license"] == "CC BY-SA 4.0"


def test_rejects_mismatched_capture_pair() -> None:
    question = {"question_id": 42, "link": "https://travel.stackexchange.com/q/42"}
    with pytest.raises(ValueError, match="matching collected_at"):
        build_thread_documents(
            capture("questions", [question]),
            capture("answers", [], collected_at="2026-07-19T12:00:00Z"),
        )


def test_rejects_answer_for_unknown_question() -> None:
    question = {"question_id": 42, "link": "https://travel.stackexchange.com/q/42"}
    answer = {"answer_id": 7, "question_id": 99}
    with pytest.raises(ValueError, match="unknown question_id"):
        build_thread_documents(capture("questions", [question]), capture("answers", [answer]))


def test_rejects_answer_without_valid_answer_id() -> None:
    question = {"question_id": 42, "link": "https://travel.stackexchange.com/q/42"}
    answer = {"question_id": 42}
    with pytest.raises(ValueError, match="positive integer answer_id"):
        build_thread_documents(capture("questions", [question]), capture("answers", [answer]))


def test_process_files_writes_one_document_per_thread_without_overwrite(tmp_path) -> None:
    question = {
        "question_id": 42,
        "title": "A title",
        "body": "<p>A question</p>",
        "link": "https://travel.stackexchange.com/q/42",
    }
    answer = {"answer_id": 7, "question_id": 42, "body": "<p>An answer</p>"}
    questions_file = tmp_path / "questions.json"
    answers_file = tmp_path / "answers.json"
    questions_file.write_text(json.dumps(capture("questions", [question])), encoding="utf-8")
    answers_file.write_text(json.dumps(capture("answers", [answer])), encoding="utf-8")

    run_dir = process_files(questions_file, answers_file, tmp_path / "cleaned")
    saved = json.loads((run_dir / "question_42.json").read_text(encoding="utf-8"))

    assert saved["question"]["text"] == "A question"
    assert saved["answers"][0]["text"] == "An answer"
    with pytest.raises(FileExistsError):
        process_files(questions_file, answers_file, tmp_path / "cleaned")


def test_processes_all_manifest_captures_and_deduplicates_questions(tmp_path) -> None:
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    question = {
        "question_id": 42,
        "title": "A title",
        "body": "<p>A question</p>",
        "link": "https://travel.stackexchange.com/q/42",
    }
    answer = {"answer_id": 7, "question_id": 42, "body": "<p>An answer</p>"}
    question_paths = ["porto/questions.json", "surfing/questions.json"]
    for relative_path in question_paths:
        path = pilot_dir / relative_path
        path.parent.mkdir()
        path.write_text(json.dumps(capture("questions", [question])), encoding="utf-8")
    answers_path = pilot_dir / "answers-1/answers.json"
    answers_path.parent.mkdir()
    answers_path.write_text(json.dumps(capture("answers", [answer])), encoding="utf-8")
    manifest_path = pilot_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "collected_at": "2026-07-18T12:00:00Z",
                "question_captures": question_paths,
                "answer_captures": ["answers-1/answers.json"],
                "unique_question_count": 1,
            }
        ),
        encoding="utf-8",
    )

    run_dir = process_manifest(manifest_path, tmp_path / "cleaned")
    saved = json.loads((run_dir / "question_42.json").read_text(encoding="utf-8"))

    assert saved["question"]["text"] == "A question"
    assert len(saved["answers"]) == 1


def test_manifest_rejects_paths_outside_pilot_directory(tmp_path) -> None:
    manifest_path = tmp_path / "pilot/manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "collected_at": "2026-07-18T12:00:00Z",
                "question_captures": ["../questions.json"],
                "answer_captures": ["answers.json"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="leaves the pilot directory"):
        load_pilot_captures(manifest_path)


def test_manifest_rejects_conflicting_duplicate_questions(tmp_path) -> None:
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    first = {"question_id": 42, "title": "First"}
    second = {"question_id": 42, "title": "Changed"}
    for name, question in (("first.json", first), ("second.json", second)):
        (pilot_dir / name).write_text(
            json.dumps(capture("questions", [question])), encoding="utf-8"
        )
    (pilot_dir / "answers.json").write_text(
        json.dumps(capture("answers", [])), encoding="utf-8"
    )
    manifest_path = pilot_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "collected_at": "2026-07-18T12:00:00Z",
                "question_captures": ["first.json", "second.json"],
                "answer_captures": ["answers.json"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting question data"):
        load_pilot_captures(manifest_path)
