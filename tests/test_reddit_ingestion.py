import json
from datetime import datetime, timezone

import pytest

from src.ingestion.reddit import listing_url, save_raw_response, validate_limit


def test_listing_url_combines_subreddits() -> None:
    assert listing_url(["r/travel", " solotravel "]) == (
        "https://oauth.reddit.com/r/travel+solotravel/hot"
    )


@pytest.mark.parametrize("limit", [0, 101])
def test_limit_must_match_reddit_listing_bounds(limit: int) -> None:
    with pytest.raises(ValueError):
        validate_limit(limit)


def test_save_preserves_raw_response(tmp_path) -> None:
    raw_response = {
        "kind": "Listing",
        "data": {"children": [{"kind": "t3", "data": {"selftext": "Cafés & sea"}}]},
    }
    collected_at = datetime(2026, 7, 18, 10, 30, tzinfo=timezone.utc)

    path = save_raw_response(raw_response, ["travel"], tmp_path, collected_at)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["response"] == raw_response
    assert saved["collection"]["collected_at"] == "2026-07-18T10:30:00Z"
    assert path.name == "reddit_posts_20260718T103000Z.json"

    with pytest.raises(FileExistsError):
        save_raw_response(raw_response, ["travel"], tmp_path, collected_at)
