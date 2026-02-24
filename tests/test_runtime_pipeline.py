"""Tests for drift_runtime.pipeline helpers."""

from dataclasses import dataclass
from drift_runtime.pipeline import deduplicate, group_by


def test_deduplicate_dicts():
    items = [
        {"address": "123 Main", "price": 100},
        {"address": "456 Oak", "price": 200},
        {"address": "123 Main", "price": 150},
    ]
    result = deduplicate(items, "address")
    assert len(result) == 2


def test_deduplicate_preserves_first():
    items = [{"id": "a", "value": 1}, {"id": "a", "value": 2}]
    result = deduplicate(items, "id")
    assert len(result) == 1
    assert result[0]["value"] == 1


def test_deduplicate_dataclasses():
    @dataclass
    class Item:
        id: str
        value: int
    items = [Item("a", 1), Item("b", 2), Item("a", 3)]
    result = deduplicate(items, "id")
    assert len(result) == 2
    assert result[0].value == 1


def test_deduplicate_empty():
    assert deduplicate([], "id") == []


def test_group_by_dicts():
    items = [
        {"city": "Austin", "name": "A"},
        {"city": "Austin", "name": "B"},
        {"city": "Denver", "name": "C"},
    ]
    result = group_by(items, "city")
    assert len(result) == 2
    austin = [g for g in result if g["key"] == "Austin"][0]
    assert len(austin["items"]) == 2
    denver = [g for g in result if g["key"] == "Denver"][0]
    assert len(denver["items"]) == 1


def test_group_by_dataclasses():
    @dataclass
    class Item:
        category: str
        name: str
    items = [Item("a", "X"), Item("b", "Y"), Item("a", "Z")]
    result = group_by(items, "category")
    assert len(result) == 2


def test_group_by_empty():
    assert group_by([], "id") == []


def test_group_by_single_group():
    items = [{"type": "x", "v": 1}, {"type": "x", "v": 2}]
    result = group_by(items, "type")
    assert len(result) == 1
    assert result[0]["key"] == "x"
    assert len(result[0]["items"]) == 2
