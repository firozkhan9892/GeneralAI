"""Tests for the collection registry."""

import pytest

from app.knowledge.collection_registry import CollectionRegistry
from app.knowledge.exceptions import (
    KnowledgeCollectionNotFoundError,
    KnowledgeDuplicateError,
)
from app.knowledge.models import CollectionMetadata, CollectionStatus


def _make_collection(
    collection_id: str = "col1",
    name: str = "Test Collection",
    namespace: str = "default",
) -> CollectionMetadata:
    return CollectionMetadata(
        collection_id=collection_id,
        name=name,
        namespace=namespace,
        status=CollectionStatus.ACTIVE,
    )


def test_add_and_get() -> None:
    reg = CollectionRegistry()
    coll = _make_collection()
    reg.add(coll)
    assert reg.get("col1").name == "Test Collection"


def test_add_duplicate_raises() -> None:
    reg = CollectionRegistry()
    reg.add(_make_collection())
    with pytest.raises(KnowledgeDuplicateError):
        reg.add(_make_collection())


def test_get_not_found() -> None:
    reg = CollectionRegistry()
    with pytest.raises(KnowledgeCollectionNotFoundError):
        reg.get("nonexistent")


def test_update() -> None:
    reg = CollectionRegistry()
    reg.add(_make_collection())
    updated = _make_collection(name="Updated")
    reg.update(updated)
    assert reg.get("col1").name == "Updated"


def test_update_not_found() -> None:
    reg = CollectionRegistry()
    with pytest.raises(KnowledgeCollectionNotFoundError):
        reg.update(_make_collection())


def test_delete() -> None:
    reg = CollectionRegistry()
    reg.add(_make_collection())
    removed = reg.delete("col1")
    assert removed.name == "Test Collection"
    assert reg.count() == 0


def test_delete_not_found() -> None:
    reg = CollectionRegistry()
    with pytest.raises(KnowledgeCollectionNotFoundError):
        reg.delete("nonexistent")


def test_list_all() -> None:
    reg = CollectionRegistry()
    reg.add(_make_collection("c1", "C1"))
    reg.add(_make_collection("c2", "C2"))
    assert len(reg.list_all()) == 2


def test_list_by_namespace() -> None:
    reg = CollectionRegistry()
    reg.add(_make_collection("c1", "C1", namespace="ns1"))
    reg.add(_make_collection("c2", "C2", namespace="ns2"))
    reg.add(_make_collection("c3", "C3", namespace="ns1"))
    assert len(reg.list_by_namespace("ns1")) == 2
    assert len(reg.list_by_namespace("ns2")) == 1


def test_count() -> None:
    reg = CollectionRegistry()
    assert reg.count() == 0
    reg.add(_make_collection())
    assert reg.count() == 1


def test_count_by_namespace() -> None:
    reg = CollectionRegistry()
    reg.add(_make_collection("c1", "C1", namespace="ns1"))
    reg.add(_make_collection("c2", "C2", namespace="ns1"))
    assert reg.count_by_namespace("ns1") == 2
    assert reg.count_by_namespace("ns2") == 0


def test_exists() -> None:
    reg = CollectionRegistry()
    assert not reg.exists("col1")
    reg.add(_make_collection())
    assert reg.exists("col1")


def test_iter() -> None:
    reg = CollectionRegistry()
    reg.add(_make_collection("c1", "C1"))
    reg.add(_make_collection("c2", "C2"))
    items = list(reg)
    assert len(items) == 2


def test_len() -> None:
    reg = CollectionRegistry()
    assert len(reg) == 0
    reg.add(_make_collection())
    assert len(reg) == 1
