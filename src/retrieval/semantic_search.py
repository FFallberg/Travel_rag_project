"""Search local travel documents with normalized embedding similarity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.local import EmbeddingModel, create_local_model, encode_texts

TAG_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "av",
    "en",
    "eller",
    "ett",
    "for",
    "from",
    "från",
    "för",
    "i",
    "in",
    "is",
    "med",
    "och",
    "of",
    "on",
    "or",
    "på",
    "the",
    "till",
    "to",
    "utan",
    "with",
    "without",
    "är",
}


@dataclass(frozen=True)
class SearchIndex:
    """Validated in-memory document records and their normalized vectors."""

    model_name: str
    document_ids: np.ndarray
    embeddings: np.ndarray
    records: tuple[dict[str, Any], ...]


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read embedding manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Embedding manifest {path} must contain a JSON object")
    return manifest


def _resolve_source_file(manifest: dict[str, Any], manifest_path: Path) -> Path:
    source_value = manifest.get("source_file")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("Embedding manifest is missing source_file")
    source_path = Path(source_value)
    if source_path.is_absolute() or source_path.exists():
        return source_path
    sibling_path = manifest_path.parent / source_path.name
    if sibling_path.exists():
        return sibling_path
    return source_path


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read retrieval documents {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} of {path} must be a JSON object")
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"Line {line_number} of {path} has an invalid document_id")
        if document_id in records:
            raise ValueError(f"Duplicate document_id in {path}: {document_id}")
        records[document_id] = record
    if not records:
        raise ValueError(f"No retrieval documents found in {path}")
    return records


def load_search_index(
    manifest_path: Path,
    documents_file: Path | None = None,
) -> SearchIndex:
    """Load and cross-validate manifest, vectors, IDs, and source documents."""
    manifest = _load_manifest(manifest_path)
    model_name = manifest.get("model")
    vectors_file = manifest.get("vectors_file")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Embedding manifest is missing model")
    if not isinstance(vectors_file, str) or not vectors_file:
        raise ValueError("Embedding manifest is missing vectors_file")

    source_path = documents_file or _resolve_source_file(manifest, manifest_path)
    expected_hash = manifest.get("source_sha256")
    try:
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"Could not read source file {source_path}: {error}") from error
    if actual_hash != expected_hash:
        raise ValueError("Retrieval document checksum does not match embedding manifest")

    vectors_path = manifest_path.parent / vectors_file
    try:
        with np.load(vectors_path, allow_pickle=False) as stored:
            document_ids = stored["document_ids"].astype(str)
            embeddings = stored["embeddings"].astype(np.float32)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"Could not load embedding vectors {vectors_path}: {error}") from error

    if document_ids.ndim != 1 or embeddings.ndim != 2:
        raise ValueError("Embedding artifact has invalid ID or vector dimensions")
    if len(document_ids) != embeddings.shape[0]:
        raise ValueError("Embedding document ID and vector counts do not match")
    if embeddings.shape[1] != manifest.get("dimensions"):
        raise ValueError("Embedding dimensions do not match manifest")
    if len(document_ids) != manifest.get("document_count"):
        raise ValueError("Embedding document count does not match manifest")
    if len(set(document_ids.tolist())) != len(document_ids):
        raise ValueError("Embedding artifact contains duplicate document IDs")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding artifact contains NaN or infinite values")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise ValueError("Embedding artifact contains non-normalized vectors")

    records_by_id = _load_records(source_path)
    if set(document_ids.tolist()) != set(records_by_id):
        raise ValueError("Embedding IDs do not match retrieval document IDs")
    ordered_records = tuple(records_by_id[document_id] for document_id in document_ids)
    return SearchIndex(model_name, document_ids, embeddings, ordered_records)


def _token_variants(value: str) -> set[str]:
    """Return lowercase tokens plus lightweight English singular variants."""
    variants: set[str] = set()
    for token in re.findall(r"\w+", value.lower()):
        if token in TAG_STOPWORDS:
            continue
        variants.add(token)
        if len(token) > 4 and token.endswith("ies"):
            variants.add(f"{token[:-3]}y")
        elif len(token) > 4 and token.endswith("es"):
            variants.add(token[:-2])
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            variants.add(token[:-1])
    return variants


def _matching_tags(query_tokens: set[str], record: dict[str, Any]) -> list[str]:
    """Return source tags with at least one token represented in the query."""
    metadata = record.get("metadata")
    tags = metadata.get("tags") if isinstance(metadata, dict) else None
    if not isinstance(tags, list):
        return []
    return [
        tag
        for tag in tags
        if isinstance(tag, str) and query_tokens.intersection(_token_variants(tag))
    ]


def search(
    index: SearchIndex,
    query: str,
    top_k: int = 5,
    model: EmbeddingModel | None = None,
    unique_threads: bool = False,
    tag_boost: float = 0.0,
) -> list[dict[str, Any]]:
    """Return the most similar records, optionally limited to one per thread."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not isinstance(tag_boost, (int, float)) or isinstance(tag_boost, bool):
        raise ValueError("tag_boost must be a non-negative finite number")
    if not np.isfinite(tag_boost) or tag_boost < 0:
        raise ValueError("tag_boost must be a non-negative finite number")
    active_model = model or create_local_model(index.model_name)
    query_vector = encode_texts(active_model, [query.strip()], batch_size=1)[0]
    if query_vector.shape[0] != index.embeddings.shape[1]:
        raise ValueError(
            f"Query embedding has {query_vector.shape[0]} dimensions; "
            f"index expects {index.embeddings.shape[1]}"
        )

    semantic_scores = index.embeddings @ query_vector
    query_tokens = _token_variants(query)
    matched_tags = [_matching_tags(query_tokens, record) for record in index.records]
    ranking_scores = semantic_scores + np.asarray(
        [len(matches) * tag_boost for matches in matched_tags], dtype=np.float32
    )
    ranked_indices = np.argsort(-ranking_scores, kind="stable")
    results: list[dict[str, Any]] = []
    seen_question_ids: set[int] = set()
    for position in ranked_indices:
        record = index.records[int(position)]
        metadata = record.get("metadata")
        question_id = metadata.get("question_id") if isinstance(metadata, dict) else None
        if unique_threads and isinstance(question_id, int) and not isinstance(question_id, bool):
            if question_id in seen_question_ids:
                continue
            seen_question_ids.add(question_id)
        results.append(
            {
                "document_id": record["document_id"],
                "score": float(ranking_scores[position]),
                "semantic_score": float(semantic_scores[position]),
                "tag_matches": matched_tags[int(position)],
                "text": record.get("text"),
                "source_url": record.get("source_url"),
                "content_license": record.get("content_license"),
                "metadata": record.get("metadata"),
            }
        )
        if len(results) == min(top_k, len(ranking_scores)):
            break
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--documents-file", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--unique-threads",
        action="store_true",
        help="Return at most one document per Stack Exchange question thread",
    )
    parser.add_argument(
        "--tag-boost",
        type=float,
        default=0.0,
        help="Score bonus for each source tag represented in the query",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        index = load_search_index(args.manifest, args.documents_file)
        results = search(
            index,
            args.query,
            args.top_k,
            unique_threads=args.unique_threads,
            tag_boost=args.tag_boost,
        )
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Semantic search failed: {error}") from error
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
