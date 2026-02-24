"""Drift runtime types.

DriftDict — dict subclass with attribute access for Drift dot notation.
ConfidentValue — wraps a value + confidence score, supports numeric comparisons.
schema_to_json_description — introspects dataclass fields for AI prompting.
parse_ai_response_to_schema — parses AI JSON responses (handles code fences).
"""

import dataclasses
import json


class DriftDict(dict):
    """A dict that supports attribute access for Drift dot notation."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No field '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


def _to_drift_dict(obj):
    """Recursively convert dicts to DriftDict and process lists."""
    if isinstance(obj, dict):
        return DriftDict({k: _to_drift_dict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_drift_dict(item) for item in obj]
    return obj


@dataclasses.dataclass
class ConfidentValue:
    """A value with an associated confidence score (0.0 to 1.0)."""
    value: object
    confidence: float

    def __repr__(self):
        return f"ConfidentValue({self.value}, confidence={self.confidence:.0%})"

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.value > other
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.value < other
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return self.value >= other
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return self.value <= other
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.value == other
        if isinstance(other, ConfidentValue):
            return self.value == other.value and self.confidence == other.confidence
        return NotImplemented

    def __hash__(self):
        return hash((self.value, self.confidence))


def schema_to_json_description(cls) -> str:
    """Produce a JSON schema description string from a dataclass for AI prompting."""
    fields = {}
    for f in dataclasses.fields(cls):
        fields[f.name] = f.type.__name__ if hasattr(f.type, "__name__") else str(f.type)
    return json.dumps(fields, indent=2)


def parse_ai_response_to_schema(response: str, schema_class):
    """Parse an AI JSON response into a dataclass instance.
    Handles markdown code fences, extra whitespace, and extra fields.
    """
    text = response.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    text = text.strip()
    data = json.loads(text)

    # Only pass fields the dataclass expects
    valid_fields = {f.name for f in dataclasses.fields(schema_class)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    return schema_class(**filtered)
