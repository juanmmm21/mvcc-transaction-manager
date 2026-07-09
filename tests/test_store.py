"""Tests directos de `InMemoryRowStore`, independientes del gestor MVCC."""

from __future__ import annotations

from mvcc_transaction_manager.models import RowId, RowVersion
from mvcc_transaction_manager.store import InMemoryRowStore


def _version(row_id: str, value: bytes | None, created_by: int, commit_seq: int) -> RowVersion:
    return RowVersion(
        row_id=RowId(row_id), value=value, created_by=created_by, commit_seq=commit_seq
    )


def test_append_and_versions_of_preserve_insertion_order() -> None:
    store = InMemoryRowStore()
    v1 = _version("x", b"a", created_by=1, commit_seq=1)
    v2 = _version("x", b"b", created_by=2, commit_seq=2)
    store.append_version(v1)
    store.append_version(v2)
    assert list(store.versions_of(RowId("x"))) == [v1, v2]


def test_versions_of_unknown_row_is_empty() -> None:
    store = InMemoryRowStore()
    assert list(store.versions_of(RowId("missing"))) == []


def test_row_ids_lists_every_row_with_at_least_one_version() -> None:
    store = InMemoryRowStore()
    store.append_version(_version("x", b"a", created_by=1, commit_seq=1))
    store.append_version(_version("y", b"b", created_by=1, commit_seq=1))
    assert set(store.row_ids()) == {RowId("x"), RowId("y")}


def test_prune_versions_replaces_chain_and_reports_removed_count() -> None:
    store = InMemoryRowStore()
    v1 = _version("x", b"a", created_by=1, commit_seq=1)
    v2 = _version("x", b"b", created_by=2, commit_seq=2)
    v3 = _version("x", b"c", created_by=3, commit_seq=3)
    for v in (v1, v2, v3):
        store.append_version(v)

    removed = store.prune_versions(RowId("x"), [v3])
    assert removed == 2
    assert list(store.versions_of(RowId("x"))) == [v3]


def test_prune_versions_with_empty_keep_removes_the_row_entirely() -> None:
    store = InMemoryRowStore()
    store.append_version(_version("x", b"a", created_by=1, commit_seq=1))

    removed = store.prune_versions(RowId("x"), [])
    assert removed == 1
    assert RowId("x") not in set(store.row_ids())
    assert list(store.versions_of(RowId("x"))) == []
