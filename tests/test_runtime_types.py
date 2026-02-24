"""Tests for drift_runtime.types module."""

import pytest
from dataclasses import dataclass
from drift_runtime.types import (
    ConfidentValue,
    schema_to_json_description,
    parse_ai_response_to_schema,
)


def test_confident_value_stores_value_and_confidence():
    cv = ConfidentValue(value=350000, confidence=0.85)
    assert cv.value == 350000
    assert cv.confidence == 0.85


def test_confident_value_gt_number():
    cv = ConfidentValue(value=350000, confidence=0.85)
    assert cv > 300000
    assert not (cv > 400000)


def test_confident_value_lt_number():
    cv = ConfidentValue(value=350000, confidence=0.85)
    assert cv < 400000
    assert not (cv < 300000)


def test_confident_value_repr():
    cv = ConfidentValue(value=100, confidence=0.92)
    r = repr(cv)
    assert "100" in r
    assert "92%" in r


def test_confident_value_ge_number():
    cv = ConfidentValue(value=100, confidence=0.5)
    assert cv >= 100
    assert cv >= 99
    assert not (cv >= 101)


def test_confident_value_le_number():
    cv = ConfidentValue(value=100, confidence=0.5)
    assert cv <= 100
    assert cv <= 101
    assert not (cv <= 99)


def test_confident_value_eq_number():
    cv = ConfidentValue(value=42, confidence=0.9)
    assert cv == 42
    assert not (cv == 43)


def test_schema_to_json_description_simple():
    @dataclass
    class Score:
        name: str
        value: int
    desc = schema_to_json_description(Score)
    assert "name" in desc
    assert "value" in desc
    assert "str" in desc
    assert "int" in desc


def test_schema_to_json_description_multiple_types():
    @dataclass
    class Analysis:
        address: str
        arv: float
        photos: list
    desc = schema_to_json_description(Analysis)
    assert "address" in desc
    assert "arv" in desc
    assert "photos" in desc


def test_parse_plain_json():
    @dataclass
    class Score:
        name: str
        value: int
    result = parse_ai_response_to_schema('{"name": "test", "value": 42}', Score)
    assert result.name == "test"
    assert result.value == 42


def test_parse_fenced_json():
    @dataclass
    class Score:
        name: str
    result = parse_ai_response_to_schema('```json\n{"name": "test"}\n```', Score)
    assert result.name == "test"


def test_parse_fenced_json_no_language_tag():
    @dataclass
    class Score:
        name: str
    result = parse_ai_response_to_schema('```\n{"name": "test"}\n```', Score)
    assert result.name == "test"


def test_parse_json_with_whitespace():
    @dataclass
    class Score:
        name: str
    result = parse_ai_response_to_schema('  \n  {"name": "test"}  \n  ', Score)
    assert result.name == "test"


def test_parse_malformed_json_raises():
    @dataclass
    class Score:
        name: str
    with pytest.raises(Exception):
        parse_ai_response_to_schema("not json at all", Score)


def test_parse_json_extra_fields_ignored():
    @dataclass
    class Score:
        name: str
    result = parse_ai_response_to_schema('{"name": "test", "extra": 99}', Score)
    assert result.name == "test"
