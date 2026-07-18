"""Collect a small raw dataset from Reddit's official OAuth API."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import requests
from dotenv import load_dotenv

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API_ROOT = "https://oauth.reddit.com"
DEFAULT_SUBREDDITS = ("travel", "solotravel")
DEFAULT_OUTPUT_DIR = Path("data/raw")
MAX_POSTS = 100


def validate_limit(limit: int) -> int:
    """Return a Reddit-compatible listing limit."""
    if not 1 <= limit <= MAX_POSTS:
        raise ValueError(f"limit must be between 1 and {MAX_POSTS}")
    return limit


def listing_url(subreddits: Sequence[str]) -> str:
    """Build the OAuth URL for a combined subreddit hot listing."""
    cleaned = [name.strip().removeprefix("r/") for name in subreddits if name.strip()]
    if not cleaned:
        raise ValueError("at least one subreddit is required")
    if any("/" in name or "+" in name for name in cleaned):
        raise ValueError("subreddit names must not contain '/' or '+'")
    return f"{API_ROOT}/r/{'+'.join(cleaned)}/hot"


def get_access_token(
    client_id: str,
    client_secret: str,
    user_agent: str,
    session: requests.Session,
) -> str:
    """Request an application-only OAuth access token."""
    response = session.post(
        TOKEN_URL,
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": user_agent},
        timeout=30,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Reddit token response did not contain an access token")
    return str(token)


def fetch_posts(
    subreddits: Sequence[str],
    limit: int,
    client_id: str,
    client_secret: str,
    user_agent: str,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Return Reddit's listing response without transforming its contents."""
    validate_limit(limit)
    http = session or requests.Session()
    token = get_access_token(client_id, client_secret, user_agent, http)
    response = http.get(
        listing_url(subreddits),
        params={"limit": limit, "raw_json": 1},
        headers={"Authorization": f"bearer {token}", "User-Agent": user_agent},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Reddit listing response was not a JSON object")
    return payload


def save_raw_response(
    response: dict[str, Any],
    subreddits: Sequence[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    collected_at: datetime | None = None,
) -> Path:
    """Save an unmodified API response plus separate collection metadata."""
    timestamp = collected_at or datetime.now(timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"reddit_posts_{timestamp:%Y%m%dT%H%M%SZ}.json"
    document = {
        "collection": {
            "collected_at": timestamp.isoformat().replace("+00:00", "Z"),
            "subreddits": list(subreddits),
            "endpoint": "hot",
        },
        "response": response,
    }
    with path.open("x", encoding="utf-8") as output:
        json.dump(document, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25, help="Number of posts (1-100)")
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=list(DEFAULT_SUBREDDITS),
        help="Subreddit names to combine into one listing",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")
    variables = {
        "REDDIT_CLIENT_ID": client_id,
        "REDDIT_CLIENT_SECRET": client_secret,
        "REDDIT_USER_AGENT": user_agent,
    }
    missing = [name for name, value in variables.items() if not value]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}")

    try:
        response = fetch_posts(
            args.subreddits,
            args.limit,
            client_id=client_id or "",
            client_secret=client_secret or "",
            user_agent=user_agent or "",
        )
        path = save_raw_response(response, args.subreddits, args.output_dir)
    except (requests.RequestException, RuntimeError, ValueError, FileExistsError) as error:
        raise SystemExit(f"Reddit ingestion failed: {error}") from error
    count = len(response.get("data", {}).get("children", []))
    print(f"Saved {count} untouched Reddit posts to {path}")


if __name__ == "__main__":
    main()
