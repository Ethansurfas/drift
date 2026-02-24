"""Tests for drift_runtime.ai module.

ALL tests mock DriftAI._call_model — no real API calls.
"""

from unittest.mock import patch, MagicMock, mock_open
from dataclasses import dataclass
from drift_runtime.ai import DriftAI


@dataclass
class TestSchema:
    name: str
    score: float


def test_drift_ai_creates():
    ai = DriftAI()
    assert ai is not None


def test_ask_simple():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="42"):
        result = ai.ask("What is the meaning of life?")
    assert result == "42"


def test_ask_with_context():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="yes") as mock:
        result = ai.ask("Is this good?", context={"data": "test"})
    assert result == "yes"
    call_args = mock.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = [m for m in messages if m["role"] == "user"][0]
    assert "test" in user_msg["content"]


def test_ask_with_schema():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value='{"name": "test", "score": 95.0}'):
        result = ai.ask("Analyze this", schema=TestSchema)
    assert isinstance(result, TestSchema)
    assert result.name == "test"
    assert result.score == 95.0


def test_ask_with_schema_fenced_response():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value='```json\n{"name": "test", "score": 80.0}\n```'):
        result = ai.ask("Analyze this", schema=TestSchema)
    assert isinstance(result, TestSchema)
    assert result.name == "test"


def test_classify():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="urgent"):
        result = ai.classify("Server is down!", categories=["urgent", "routine", "spam"])
    assert result == "urgent"


def test_classify_strips_whitespace():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="  urgent  \n"):
        result = ai.classify("Server is down!", categories=["urgent", "routine", "spam"])
    assert result == "urgent"


def test_classify_retries_on_invalid():
    ai = DriftAI()
    with patch.object(ai, "_call_model", side_effect=["invalid_category", "urgent"]):
        result = ai.classify("Server is down!", categories=["urgent", "routine", "spam"])
    assert result == "urgent"


def test_embed():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="[0.1, 0.2, 0.3]"):
        result = ai.embed("hello world")
    assert isinstance(result, list)
    assert len(result) == 3


def test_see():
    ai = DriftAI()
    fake_image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    m = mock_open(read_data=fake_image_data)
    with patch("builtins.open", m):
        with patch.object(ai, "_call_model", return_value="A red house with a green roof"):
            result = ai.see("photo.jpg", "Describe this image")
    assert "red house" in result


def test_predict_returns_confident_value():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value='{"value": 350000, "confidence": 0.85}'):
        result = ai.predict("Estimate the ARV")
    from drift_runtime.types import ConfidentValue
    assert isinstance(result, ConfidentValue)
    assert result.value == 350000
    assert result.confidence == 0.85


def test_enrich_items():
    ai = DriftAI()
    items = [{"name": "A"}, {"name": "B"}]
    with patch.object(ai, "_call_model", side_effect=[
        '{"summary": "Great A"}',
        '{"summary": "Great B"}',
    ]):
        result = ai.enrich(items, "Add a summary")
    assert result[0]["summary"] == "Great A"
    assert result[1]["summary"] == "Great B"


def test_enrich_empty_list():
    ai = DriftAI()
    result = ai.enrich([], "Add a summary")
    assert result == []


def test_score_items():
    ai = DriftAI()
    items = [{"name": "A"}, {"name": "B"}]
    with patch.object(ai, "_call_model", side_effect=["85", "42"]):
        result = ai.score(items, "Rate quality 1-100")
    assert result[0]["score"] == 85
    assert result[1]["score"] == 42


def test_score_empty_list():
    ai = DriftAI()
    result = ai.score([], "Rate quality")
    assert result == []
