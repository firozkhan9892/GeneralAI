"""Tests for the namespace registry."""

import pytest

from app.knowledge.constants import DEFAULT_NAMESPACE
from app.knowledge.exceptions import (
    KnowledgeDuplicateError,
    KnowledgeNamespaceNotFoundError,
    KnowledgeValidationError,
)
from app.knowledge.namespace_registry import NamespaceRegistry
from app.knowledge.models import NamespaceMetadata


def _make_namespace(name: str = "team-a") -> NamespaceMetadata:
    return NamespaceMetadata(name=name, description=f"Namespace {name}")


def test_default_namespace_exists() -> None:
    reg = NamespaceRegistry()
    assert reg.exists(DEFAULT_NAMESPACE)
    ns = reg.get(DEFAULT_NAMESPACE)
    assert ns.name == DEFAULT_NAMESPACE


def test_add_and_get() -> None:
    reg = NamespaceRegistry()
    ns = _make_namespace("team-b")
    reg.add(ns)
    assert reg.get("team-b").description == "Namespace team-b"


def test_add_duplicate_raises() -> None:
    reg = NamespaceRegistry()
    reg.add(_make_namespace("team-c"))
    with pytest.raises(KnowledgeDuplicateError):
        reg.add(_make_namespace("team-c"))


def test_get_not_found() -> None:
    reg = NamespaceRegistry()
    with pytest.raises(KnowledgeNamespaceNotFoundError):
        reg.get("nonexistent")


def test_update() -> None:
    reg = NamespaceRegistry()
    reg.add(_make_namespace("team-d"))
    updated = NamespaceMetadata(name="team-d", description="Updated")
    reg.update(updated)
    assert reg.get("team-d").description == "Updated"


def test_update_not_found() -> None:
    reg = NamespaceRegistry()
    with pytest.raises(KnowledgeNamespaceNotFoundError):
        reg.update(_make_namespace("nonexistent"))


def test_delete() -> None:
    reg = NamespaceRegistry()
    reg.add(_make_namespace("team-e"))
    removed = reg.delete("team-e")
    assert removed.name == "team-e"
    assert not reg.exists("team-e")


def test_delete_default_raises() -> None:
    reg = NamespaceRegistry()
    with pytest.raises(KnowledgeValidationError):
        reg.delete(DEFAULT_NAMESPACE)


def test_delete_not_found() -> None:
    reg = NamespaceRegistry()
    with pytest.raises(KnowledgeNamespaceNotFoundError):
        reg.delete("nonexistent")


def test_list_all() -> None:
    reg = NamespaceRegistry()
    reg.add(_make_namespace("a"))
    reg.add(_make_namespace("b"))
    all_ns = reg.list_all()
    assert len(all_ns) >= 3  # default + a + b


def test_count() -> None:
    reg = NamespaceRegistry()
    initial = reg.count()
    assert initial >= 1  # default
    reg.add(_make_namespace("x"))
    assert reg.count() == initial + 1


def test_exists() -> None:
    reg = NamespaceRegistry()
    assert reg.exists(DEFAULT_NAMESPACE)
    assert not reg.exists("nope")


def test_iter() -> None:
    reg = NamespaceRegistry()
    reg.add(_make_namespace("y"))
    items = list(reg)
    assert len(items) >= 2


def test_len() -> None:
    reg = NamespaceRegistry()
    assert len(reg) >= 1
