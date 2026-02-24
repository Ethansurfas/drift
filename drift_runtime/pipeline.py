"""Drift pipeline helper functions.

Used by transpiled pipeline code for deduplicate and group_by stages.
"""

from drift_runtime.types import _to_drift_dict


def deduplicate(items: list, key: str) -> list:
    """Remove duplicates from a list by a key field. Keeps the first occurrence."""
    seen = {}
    for item in items:
        k = item[key] if isinstance(item, dict) else getattr(item, key)
        if k not in seen:
            seen[k] = item
    return list(seen.values())


def group_by(items: list, key: str) -> list:
    """Group a list of items by a key field. Returns list of {key, items} dicts."""
    groups = {}
    for item in items:
        k = item[key] if isinstance(item, dict) else getattr(item, key)
        groups.setdefault(k, []).append(item)
    return [_to_drift_dict({"key": k, "items": v}) for k, v in groups.items()]
