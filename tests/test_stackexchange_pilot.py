import json
from datetime import datetime, timezone

import pytest

from src.ingestion.stackexchange_pilot import (
    batches,
    collect_pilot,
    load_pilot_config,
    unique_question_ids,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return FakeResponse(next(self.payloads))


def test_loads_valid_config_and_rejects_duplicate_names(tmp_path) -> None:
    path = tmp_path / "pilot.json"
    path.write_text(
        json.dumps({"searches": [{"name": "porto", "query": "Porto"}]}),
        encoding="utf-8",
    )
    assert load_pilot_config(path)["searches"][0]["name"] == "porto"

    path.write_text(
        json.dumps({"searches": [{"name": "porto", "query": "Porto"}] * 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate search name"):
        load_pilot_config(path)


def test_deduplicates_ids_in_first_seen_order_and_batches() -> None:
    responses = [
        {"items": [{"question_id": 2}, {"question_id": 1}]},
        {"items": [{"question_id": 1}, {"question_id": 3}]},
    ]
    assert unique_question_ids(responses) == [2, 1, 3]
    assert batches([1, 2, 3], size=2) == [[1, 2], [3]]


def test_collect_saves_untouched_searches_and_fetches_unique_answers(tmp_path) -> None:
    config = {
        "limit_per_search": 10,
        "minimum_answers": 1,
        "searches": [
            {"name": "porto", "query": "Porto", "tags": ["portugal"]},
            {"name": "surfing", "query": "surfing"},
        ],
    }
    first = {"items": [{"question_id": 1}, {"question_id": 2}]}
    second = {"items": [{"question_id": 2}, {"question_id": 3}]}
    answers = {"items": [{"answer_id": 10, "question_id": 1}]}
    session = FakeSession([first, second, answers])

    manifest_path = collect_pilot(
        config,
        tmp_path,
        session=session,
        collected_at=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["unique_question_count"] == 3
    assert manifest["duplicate_question_count"] == 1
    assert len(manifest["question_captures"]) == 2
    assert len(manifest["answer_captures"]) == 1
    first_capture = json.loads(
        (manifest_path.parent / manifest["question_captures"][0]).read_text(encoding="utf-8")
    )
    assert first_capture["response"] == first
    answer_url, _ = session.requests[-1]
    assert answer_url.endswith("/questions/1;2;3/answers")
