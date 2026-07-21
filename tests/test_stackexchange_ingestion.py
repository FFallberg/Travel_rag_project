import json
from datetime import datetime, timezone

import pytest

from src.ingestion.stackexchange import (
    QUESTIONS_URL,
    SEARCH_URL,
    answers_url,
    extract_question_ids,
    fetch_answers,
    fetch_questions,
    fetch_targeted_questions,
    normalize_tags,
    save_raw_response,
    validate_limit,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.request = None

    def get(self, url, **kwargs):
        self.request = (url, kwargs)
        return FakeResponse(self.payload)


@pytest.mark.parametrize("limit", [0, 101])
def test_limit_matches_stack_exchange_page_bounds(limit: int) -> None:
    with pytest.raises(ValueError):
        validate_limit(limit)


def test_fetch_uses_travel_site_and_preserves_response() -> None:
    raw_response = {
        "items": [{"question_id": 42, "content_license": "CC BY-SA 4.0"}],
        "quota_remaining": 299,
    }
    session = FakeSession(raw_response)

    result = fetch_questions(12, sort="activity", api_key="test-key", session=session)

    assert result is raw_response
    url, kwargs = session.request
    assert url == QUESTIONS_URL
    assert kwargs["params"] == {
        "site": "travel",
        "pagesize": 12,
        "page": 1,
        "order": "desc",
        "sort": "activity",
        "filter": "withbody",
        "key": "test-key",
    }
    assert kwargs["timeout"] == 30


def test_targeted_search_uses_advanced_endpoint_and_public_metadata() -> None:
    raw_response = {"items": [{"question_id": 42}], "quota_remaining": 290}
    session = FakeSession(raw_response)

    result, request_params = fetch_targeted_questions(
        " Porto swimming ",
        ["portugal", "beaches"],
        10,
        api_key="secret-key",
        session=session,
    )

    assert result is raw_response
    url, kwargs = session.request
    assert url == SEARCH_URL
    assert kwargs["params"] == {
        "site": "travel",
        "pagesize": 10,
        "page": 1,
        "order": "desc",
        "sort": "relevance",
        "answers": 1,
        "filter": "withbody",
        "q": "Porto swimming",
        "tagged": "portugal;beaches",
        "key": "secret-key",
    }
    assert "key" not in request_params
    assert request_params == {key: value for key, value in kwargs["params"].items() if key != "key"}


def test_targeted_search_requires_query_or_tags() -> None:
    with pytest.raises(ValueError, match="requires a query"):
        fetch_targeted_questions(None, [], 10, session=FakeSession({}))


def test_tags_are_limited_clean_and_unique() -> None:
    assert normalize_tags([" portugal ", "beaches"]) == ["portugal", "beaches"]
    with pytest.raises(ValueError, match="at most 5"):
        normalize_tags(["a", "b", "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="semicolons"):
        normalize_tags(["portugal;beaches"])
    with pytest.raises(ValueError, match="duplicates"):
        normalize_tags(["portugal", "portugal"])


def test_extract_ids_and_fetch_answers_as_one_batch() -> None:
    questions = {
        "items": [
            {"question_id": 42},
            {"question_id": 99},
            {"question_id": 42},
            {"not_a_question": True},
        ]
    }
    raw_answers = {
        "items": [
            {"answer_id": 7, "question_id": 42, "content_license": "CC BY-SA 4.0"}
        ]
    }
    session = FakeSession(raw_answers)

    question_ids = extract_question_ids(questions)
    result = fetch_answers(question_ids, limit=50, session=session)

    assert question_ids == [42, 99]
    assert result is raw_answers
    url, kwargs = session.request
    assert url == f"{QUESTIONS_URL}/42;99/answers"
    assert kwargs["params"] == {
        "site": "travel",
        "pagesize": 50,
        "page": 1,
        "order": "desc",
        "sort": "votes",
        "filter": "withbody",
    }


def test_answers_url_requires_valid_ids() -> None:
    with pytest.raises(ValueError):
        answers_url([])
    with pytest.raises(ValueError):
        answers_url([-1])


def test_save_preserves_raw_response_and_will_not_overwrite(tmp_path) -> None:
    raw_response = {
        "items": [
            {
                "question_id": 42,
                "body": "Where should I swim?",
                "link": "https://travel.stackexchange.com/q/42",
                "content_license": "CC BY-SA 4.0",
            }
        ]
    }
    collected_at = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)

    path = save_raw_response(raw_response, tmp_path, collected_at)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["response"] == raw_response
    assert saved["collection"]["collected_at"] == "2026-07-18T12:00:00Z"
    assert path.name == "stackexchange_questions_20260718T120000Z.json"
    with pytest.raises(FileExistsError):
        save_raw_response(raw_response, tmp_path, collected_at)


def test_save_answers_uses_separate_matching_filename(tmp_path) -> None:
    raw_response = {"items": [{"answer_id": 7, "question_id": 42}]}
    collected_at = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)

    path = save_raw_response(
        raw_response,
        tmp_path,
        collected_at,
        resource="answers",
        endpoint=f"{QUESTIONS_URL}/42/answers",
    )
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["response"] == raw_response
    assert saved["collection"]["resource"] == "answers"
    assert saved["collection"]["endpoint"] == f"{QUESTIONS_URL}/42/answers"
    assert path.name == "stackexchange_answers_20260718T120000Z.json"


def test_save_targeted_search_keeps_request_separate_from_response(tmp_path) -> None:
    raw_response = {"items": [{"question_id": 42}]}
    request = {"q": "Porto swimming", "tagged": "portugal"}

    path = save_raw_response(
        raw_response,
        tmp_path,
        datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc),
        endpoint=SEARCH_URL,
        request_params=request,
    )
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["response"] == raw_response
    assert saved["collection"]["endpoint"] == SEARCH_URL
    assert saved["collection"]["request"] == request
