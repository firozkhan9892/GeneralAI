"""Metadata filter evaluation.

Evaluates :class:`MetadataFilter` instances against chunk metadata
dictionaries.  Filters use conjunctive (AND) semantics: all filters
must pass for a chunk to be included.

The supported operators are: ``eq``, ``neq``, ``in``, ``not_in``,
``gt``, ``gte``, ``lt``, ``lte``, ``exists``, ``contains``.
"""

from __future__ import annotations

from typing import Any

from app.knowledge.models import MetadataFilter


def evaluate_filter(metadata: dict[str, Any], f: MetadataFilter) -> bool:
    """Return ``True`` if *metadata* satisfies filter *f*.

    Args:
        metadata: The chunk metadata dictionary.
        f: The filter to evaluate.

    Returns:
        Whether the filter condition is met.
    """
    field_value = metadata.get(f.field)
    op = f.op

    if op == "eq":
        return field_value == f.value
    if op == "neq":
        return field_value != f.value
    if op == "in":
        return (
            field_value in f.value if isinstance(f.value, (list, tuple, set)) else False
        )
    if op == "not_in":
        return (
            field_value not in f.value
            if isinstance(f.value, (list, tuple, set))
            else True
        )
    if op == "gt":
        return _compare_numeric(field_value, f.value, lambda a, b: a > b)
    if op == "gte":
        return _compare_numeric(field_value, f.value, lambda a, b: a >= b)
    if op == "lt":
        return _compare_numeric(field_value, f.value, lambda a, b: a < b)
    if op == "lte":
        return _compare_numeric(field_value, f.value, lambda a, b: a <= b)
    if op == "exists":
        return (field_value is not None) == bool(f.value)
    if op == "contains":
        if isinstance(field_value, str) and isinstance(f.value, str):
            return f.value in field_value
        if isinstance(field_value, (list, tuple)):
            return f.value in field_value
        return False

    return False


def evaluate_filters(
    metadata: dict[str, Any], filters: tuple[MetadataFilter, ...]
) -> bool:
    """Return ``True`` if *metadata* passes **all** *filters*.

    Conjunctive (AND) semantics — an empty filter tuple always passes.
    """
    return all(evaluate_filter(metadata, f) for f in filters)


def _compare_numeric(a: Any, b: Any, op: Any) -> bool:  # noqa: N802
    """Compare two values, returning ``False`` if either is ``None``."""
    if a is None or b is None:
        return False
    try:
        return op(a, b)
    except TypeError:
        return False
