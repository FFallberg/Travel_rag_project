import numpy as np
import pytest

from src.generation.recommendation import OllamaTextGenerator, build_prompt, recommend
from src.retrieval.semantic_search import SearchIndex


class QueryModel:
    def __init__(self, vector):
        self.vector = np.asarray([vector], dtype=np.float32)

    def encode(self, sentences, **kwargs):
        return self.vector


class RecordingGenerator:
    def __init__(self, answer="Besök viken för bad. [Källa 1]"):
        self.answer = answer
        self.calls = []

    def generate(self, instructions, prompt):
        self.calls.append((instructions, prompt))
        return self.answer


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"role": "assistant", "content": "Lokalt svar"}}


def make_index():
    records = (
        {
            "document_id": "doc-water",
            "text": "The sheltered bay is suitable for swimming.",
            "source_url": "https://example.com/water",
            "content_license": "CC BY-SA 4.0",
            "metadata": {
                "question_id": 10,
                "author": "Ada",
                "tags": ["beaches"],
                "content_type": "answer",
            },
        },
        {
            "document_id": "doc-train",
            "text": "The night train is convenient.",
            "source_url": "https://example.com/train",
            "content_license": "CC BY-SA 4.0",
            "metadata": {
                "question_id": 20,
                "author": "Lin",
                "tags": ["trains"],
                "content_type": "answer",
            },
        },
    )
    return SearchIndex(
        "fake-model",
        np.asarray(["doc-water", "doc-train"]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        records,
    )


def test_generates_answer_from_only_results_above_threshold() -> None:
    generator = RecordingGenerator()
    result = recommend(
        make_index(),
        "Jag vill bada",
        generator,
        top_k=2,
        min_score=0.7,
        embedding_model=QueryModel([0.8, 0.6]),
        tag_boost=0,
    )

    assert result["abstained"] is False
    assert result["answer"] == "Besök viken för bad. [Källa 1]"
    assert [source["document_id"] for source in result["sources"]] == ["doc-water"]
    assert result["sources"][0]["content_license"] == "CC BY-SA 4.0"
    assert "doc-water" in generator.calls[0][1]
    assert "doc-train" not in generator.calls[0][1]


def test_abstains_without_calling_generator_when_evidence_is_weak() -> None:
    generator = RecordingGenerator()
    result = recommend(
        make_index(),
        "orelaterad fråga",
        generator,
        min_score=0.9,
        embedding_model=QueryModel([0.6, 0.8]),
        tag_boost=0,
    )

    assert result["abstained"] is True
    assert result["sources"] == []
    assert generator.calls == []


def test_prompt_numbers_sources_and_rejects_empty_input() -> None:
    results = [
        {
            "document_id": "doc-1",
            "text": "Evidence",
            "source_url": "https://example.com/1",
        }
    ]
    prompt = build_prompt("Bad?", results)

    assert "[Källa 1]" in prompt
    assert "Bad?" in prompt
    assert "Evidence" in prompt
    with pytest.raises(ValueError):
        build_prompt("Bad?", [])


def test_rejects_empty_generator_response() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        recommend(
            make_index(),
            "bad",
            RecordingGenerator("  "),
            min_score=0.5,
            embedding_model=QueryModel([1.0, 0.0]),
            tag_boost=0,
        )


@pytest.mark.parametrize(
    "answer",
    ["Svar utan hänvisning.", "Fel källa. [Källa 2]"],
)
def test_rejects_missing_or_unknown_citations(answer) -> None:
    with pytest.raises(RuntimeError):
        recommend(
            make_index(),
            "bad",
            RecordingGenerator(answer),
            min_score=0.9,
            embedding_model=QueryModel([1.0, 0.0]),
            tag_boost=0,
        )


@pytest.mark.parametrize("min_score", [float("nan"), float("inf"), True])
def test_rejects_invalid_min_score(min_score) -> None:
    with pytest.raises(ValueError, match="min_score"):
        recommend(
            make_index(),
            "bad",
            RecordingGenerator(),
            min_score=min_score,
            embedding_model=QueryModel([1.0, 0.0]),
        )


def test_ollama_generator_calls_local_chat_api(monkeypatch) -> None:
    call = {}

    def fake_post(url, **kwargs):
        call.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr("src.generation.recommendation.requests.post", fake_post)

    answer = OllamaTextGenerator(
        model="local-model",
        base_url="http://localhost:11434/",
    ).generate("System", "Prompt")

    assert answer == "Lokalt svar"
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["json"]["model"] == "local-model"
    assert call["json"]["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Prompt"},
    ]
    assert call["json"]["stream"] is False
