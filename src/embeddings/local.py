"""Generate local multilingual embeddings for retrieval documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OUTPUT_DIR = Path("data/processed")


class EmbeddingModel(Protocol):
    """Minimal interface implemented by SentenceTransformer and test doubles."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any:
        """Encode text into a two-dimensional numeric array."""


def load_retrieval_records(path: Path) -> tuple[list[str], list[str]]:
    """Load unique document IDs and non-empty text from retrieval JSONL."""
    document_ids: list[str] = []
    texts: list[str] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read retrieval file {path}: {error}") from error

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
        text = record.get("text")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"Line {line_number} of {path} has an invalid document_id")
        if document_id in seen_ids:
            raise ValueError(f"Duplicate document_id in {path}: {document_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Document {document_id} has empty text")
        seen_ids.add(document_id)
        document_ids.append(document_id)
        texts.append(text.strip())

    if not document_ids:
        raise ValueError(f"No retrieval documents found in {path}")
    return document_ids, texts


def encode_texts(
    model: EmbeddingModel,
    texts: Sequence[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Generate normalized finite float32 vectors and validate their shape."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    vectors = np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
        raise ValueError(
            f"Embedding model returned shape {vectors.shape}; expected ({len(texts)}, dimensions)"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embedding model did not return normalized vectors")
    return vectors


def create_local_model(model_name: str = DEFAULT_MODEL) -> EmbeddingModel:
    """Load a SentenceTransformer model only when real embeddings are requested."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "sentence-transformers is not installed; install requirements.txt first"
        ) from error
    return SentenceTransformer(model_name)


def write_embedding_artifacts(
    document_ids: Sequence[str],
    vectors: np.ndarray,
    source_file: Path,
    model_name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    created_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Write vectors plus a reproducibility manifest without overwriting files."""
    if len(document_ids) != vectors.shape[0]:
        raise ValueError("Document ID and vector counts do not match")
    timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"stackexchange_embeddings_{timestamp:%Y%m%dT%H%M%SZ}"
    vectors_path = output_dir / f"{stem}.npz"
    manifest_path = output_dir / f"{stem}.json"
    if vectors_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Embedding output already exists for {stem}")

    with vectors_path.open("xb") as output:
        np.savez_compressed(
            output,
            document_ids=np.asarray(document_ids, dtype=str),
            embeddings=vectors.astype(np.float32, copy=False),
        )
    manifest = {
        "created_at": timestamp.isoformat().replace("+00:00", "Z"),
        "model": model_name,
        "model_library": "sentence-transformers",
        "normalized": True,
        "dtype": "float32",
        "dimensions": int(vectors.shape[1]),
        "document_count": len(document_ids),
        "source_file": str(source_file),
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "vectors_file": vectors_path.name,
    }
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return vectors_path, manifest_path


def generate_embeddings(
    input_file: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    model: EmbeddingModel | None = None,
) -> tuple[Path, Path]:
    """Load retrieval records, embed their text, and persist mapped vectors."""
    document_ids, texts = load_retrieval_records(input_file)
    active_model = model or create_local_model(model_name)
    vectors = encode_texts(active_model, texts, batch_size)
    return write_embedding_artifacts(
        document_ids,
        vectors,
        input_file,
        model_name,
        output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        vectors_path, manifest_path = generate_embeddings(
            input_file=args.input_file,
            output_dir=args.output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    except (ValueError, FileExistsError, RuntimeError, OSError) as error:
        raise SystemExit(f"Embedding generation failed: {error}") from error
    print(f"Saved embeddings to {vectors_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
