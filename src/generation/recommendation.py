"""Generate travel recommendations grounded in retrieved source documents."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
import requests

from src.embeddings.local import EmbeddingModel
from src.retrieval.semantic_search import SearchIndex, load_search_index, search

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MIN_SCORE = 0.6


class TextGenerator(Protocol):
    """Minimal interface for an LLM provider or a test double."""

    def generate(self, instructions: str, prompt: str) -> str:
        """Return generated text for the supplied instructions and prompt."""


@dataclass(frozen=True)
class OllamaTextGenerator:
    """Generate text with a model served by the local Ollama API."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_OLLAMA_URL
    timeout: float = 120.0

    def generate(self, instructions: str, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(f"Ollama request failed: {error}") from error
        message = payload.get("message") if isinstance(payload, dict) else None
        answer = message.get("content") if isinstance(message, dict) else None
        if not isinstance(answer, str):
            raise RuntimeError("Ollama returned an invalid response")
        answer = answer.strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response")
        return answer


def _source(result: dict[str, Any], number: int) -> dict[str, Any]:
    metadata = result.get("metadata")
    safe_metadata = metadata if isinstance(metadata, dict) else {}
    return {
        "number": number,
        "document_id": result["document_id"],
        "source_url": result.get("source_url"),
        "content_license": result.get("content_license"),
        "author": safe_metadata.get("author"),
        "question_id": safe_metadata.get("question_id"),
        "score": result["score"],
    }


def build_prompt(query: str, results: list[dict[str, Any]]) -> str:
    """Build a numbered evidence prompt from retrieval results."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not results:
        raise ValueError("at least one retrieval result is required")

    evidence = []
    for number, result in enumerate(results, start=1):
        source_text = result.get("text")
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError(f"Retrieved document {result.get('document_id')} has empty text")
        evidence.append(
            f"[Källa {number}]\n"
            f"Dokument-ID: {result['document_id']}\n"
            f"URL: {result.get('source_url')}\n"
            f"Text:\n{source_text.strip()}"
        )
    return f"Användarens fråga:\n{query.strip()}\n\nHämtat underlag:\n\n" + "\n\n".join(evidence)


def _validate_citations(answer: str, source_count: int) -> None:
    """Require at least one citation and reject references to absent sources."""
    citations = [int(value) for value in re.findall(r"\[Källa (\d+)\]", answer)]
    if not citations:
        raise RuntimeError("Generated answer does not cite any retrieved source")
    invalid = sorted({number for number in citations if number < 1 or number > source_count})
    if invalid:
        raise RuntimeError(
            "Generated answer cites unavailable sources: "
            + ", ".join(str(number) for number in invalid)
        )


INSTRUCTIONS = """Du är en försiktig reseassistent.
Svara på samma språk som användaren. Använd endast det hämtade underlaget för
konkreta påståenden och rekommendationer. Hänvisa direkt efter varje relevant
påstående med [Källa N]. Hitta inte på destinationer, fakta eller källor.
Förklara kort hur rekommendationen matchar önskemålen och nämn tydligt viktiga
begränsningar eller osäkerheter i underlaget. Om källorna inte räcker för en
säker slutsats, säg det. Lägg inte till en separat länklista; den bifogas av
systemet."""


def recommend(
    index: SearchIndex,
    query: str,
    generator: TextGenerator,
    *,
    top_k: int = 3,
    min_score: float = DEFAULT_MIN_SCORE,
    embedding_model: EmbeddingModel | None = None,
    unique_threads: bool = True,
    tag_boost: float = 0.05,
    answers_only: bool = True,
) -> dict[str, Any]:
    """Retrieve evidence and generate an answer, or abstain below the threshold."""
    if (
        not isinstance(min_score, (int, float))
        or isinstance(min_score, bool)
        or not math.isfinite(min_score)
    ):
        raise ValueError("min_score must be a finite number")
    results = search(
        index,
        query,
        top_k=top_k,
        model=embedding_model,
        unique_threads=unique_threads,
        tag_boost=tag_boost,
        answers_only=answers_only,
    )
    passing = [result for result in results if result["score"] >= min_score]
    sources = [_source(result, number) for number, result in enumerate(passing, start=1)]
    parameters = {
        "top_k": top_k,
        "min_score": min_score,
        "unique_threads": unique_threads,
        "tag_boost": tag_boost,
        "answers_only": answers_only,
    }
    if not passing:
        return {
            "answer": "Underlaget är för svagt för att ge en källstödd rekommendation.",
            "abstained": True,
            "sources": [],
            "retrieval": parameters,
        }
    answer = generator.generate(INSTRUCTIONS, build_prompt(query, passing))
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Text generator returned an empty response")
    _validate_citations(answer, len(sources))
    return {
        "answer": answer.strip(),
        "abstained": False,
        "sources": sources,
        "retrieval": parameters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--documents-file", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--tag-boost", type=float, default=0.05)
    parser.add_argument("--allow-duplicate-threads", action="store_true")
    parser.add_argument(
        "--include-questions",
        action="store_true",
        help="Allow question documents to be used as recommendation evidence",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        result = recommend(
            load_search_index(args.manifest, args.documents_file),
            args.query,
            OllamaTextGenerator(model=args.model, base_url=args.ollama_url),
            top_k=args.top_k,
            min_score=args.min_score,
            unique_threads=not args.allow_duplicate_threads,
            tag_boost=args.tag_boost,
            answers_only=not args.include_questions,
        )
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Recommendation generation failed: {error}") from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
