"""Seed del contador de commits al reabrir un store persistido, y camino
rápido de `scan` sobre `BulkRowStore` (equivalencia exacta con el camino
fila a fila)."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

import pytest

from mvcc_transaction_manager import (
    BulkRowStore,
    InMemoryRowStore,
    IsolationLevel,
    MVCCTransactionManager,
    RowId,
    RowVersion,
)


class LegacyOnlyStore:
    """Envoltorio que expone SOLO los métodos del `RowStore` original:
    fuerza el camino fila a fila de `scan` para poder compararlo contra el
    camino masivo sobre el mismo contenido."""

    def __init__(self, inner: InMemoryRowStore) -> None:
        self._inner = inner

    def append_version(self, version: RowVersion) -> None:
        self._inner.append_version(version)

    def versions_of(self, row_id: RowId) -> Sequence[RowVersion]:
        return self._inner.versions_of(row_id)

    def row_ids(self) -> Iterator[RowId]:
        return self._inner.row_ids()

    def prune_versions(self, row_id: RowId, keep: Sequence[RowVersion]) -> int:
        return self._inner.prune_versions(row_id, keep)


class TestConstructorSeed:
    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValueError):
            MVCCTransactionManager(InMemoryRowStore(), last_committed_seq=-1)

    def test_reopened_store_data_visible_with_seed(self) -> None:
        store = InMemoryRowStore()
        first = MVCCTransactionManager(store)
        txn = first.begin()
        first.put(txn, RowId("fila"), b"valor")
        max_seq = int(first.commit(txn))

        # Sin seed, un gestor nuevo no vería nada del store persistido
        # (todo commit_seq quedaría por encima de su horizonte 0).
        blind = MVCCTransactionManager(store)
        blind_txn = blind.begin(IsolationLevel.REPEATABLE_READ)
        assert blind.get(blind_txn, RowId("fila")) is None

        seeded = MVCCTransactionManager(store, last_committed_seq=max_seq)
        seeded_txn = seeded.begin(IsolationLevel.REPEATABLE_READ)
        assert seeded.get(seeded_txn, RowId("fila")) == b"valor"

    def test_seed_prevents_commit_seq_reuse(self) -> None:
        store = InMemoryRowStore()
        first = MVCCTransactionManager(store)
        txn = first.begin()
        first.put(txn, RowId("fila"), b"v1")
        max_seq = int(first.commit(txn))

        seeded = MVCCTransactionManager(store, last_committed_seq=max_seq)
        txn2 = seeded.begin()
        seeded.put(txn2, RowId("fila"), b"v2")
        new_seq = seeded.commit(txn2)
        assert int(new_seq) == max_seq + 1
        seqs = [int(v.commit_seq) for v in store.versions_of(RowId("fila"))]
        assert len(seqs) == len(set(seqs)), "commit_seq reutilizado"


class TestBulkScanEquivalence:
    def test_in_memory_store_satisfies_bulk_protocol(self) -> None:
        assert isinstance(InMemoryRowStore(), BulkRowStore)
        assert not isinstance(LegacyOnlyStore(InMemoryRowStore()), BulkRowStore)

    def _populated_pair(self, seed: int) -> tuple[MVCCTransactionManager, MVCCTransactionManager]:
        """Dos gestores sobre el MISMO contenido: uno con camino masivo
        (InMemoryRowStore) y otro forzado al camino fila a fila."""
        rng = random.Random(seed)
        bulk_store = InMemoryRowStore()
        manager = MVCCTransactionManager(bulk_store)
        for _ in range(rng.randint(4, 8)):
            txn = manager.begin()
            for _ in range(rng.randint(1, 6)):
                row = RowId(f"fila-{rng.randint(0, 15):03d}")
                if rng.random() < 0.25:
                    manager.delete(txn, row)
                else:
                    manager.put(txn, row, f"v{rng.random():.6f}".encode())
            if rng.random() < 0.2:
                manager.abort(txn)
            else:
                manager.commit(txn)
        legacy = MVCCTransactionManager(
            LegacyOnlyStore(bulk_store),
            last_committed_seq=manager._last_committed_seq,
        )
        return manager, legacy

    @pytest.mark.parametrize("seed", [3, 17, 4242])
    def test_bulk_and_row_by_row_scans_agree(self, seed: int) -> None:
        bulk_manager, legacy_manager = self._populated_pair(seed)
        bulk_txn = bulk_manager.begin(IsolationLevel.REPEATABLE_READ)
        legacy_txn = legacy_manager.begin(IsolationLevel.REPEATABLE_READ)
        assert bulk_manager.scan(bulk_txn) == legacy_manager.scan(legacy_txn)

    def test_bulk_scan_sees_own_uncommitted_writes_and_deletes(self) -> None:
        store = InMemoryRowStore()
        manager = MVCCTransactionManager(store)
        setup = manager.begin()
        manager.put(setup, RowId("a"), b"1")
        manager.put(setup, RowId("b"), b"2")
        manager.commit(setup)

        txn = manager.begin(IsolationLevel.REPEATABLE_READ)
        manager.put(txn, RowId("c"), b"3")  # insert propio sin commit
        manager.delete(txn, RowId("a"))  # delete propio sin commit
        assert manager.scan(txn) == [(RowId("b"), b"2"), (RowId("c"), b"3")]

    def test_bulk_scan_respects_snapshot_horizon(self) -> None:
        store = InMemoryRowStore()
        manager = MVCCTransactionManager(store)
        setup = manager.begin()
        manager.put(setup, RowId("a"), b"viejo")
        manager.commit(setup)

        reader = manager.begin(IsolationLevel.REPEATABLE_READ)
        assert manager.scan(reader) == [(RowId("a"), b"viejo")]

        writer = manager.begin()
        manager.put(writer, RowId("a"), b"nuevo")
        manager.put(writer, RowId("z"), b"fantasma")
        manager.commit(writer)

        # El snapshot del lector no ve ni la actualización ni el phantom.
        assert manager.scan(reader) == [(RowId("a"), b"viejo")]

    def test_bulk_scan_populates_read_set_with_invisible_rows(self) -> None:
        # La detección de write skew depende de que un scan registre en el
        # read_set también las filas conocidas pero no visibles; el camino
        # masivo debe conservar exactamente esa semántica.
        store = InMemoryRowStore()
        manager = MVCCTransactionManager(store)
        setup = manager.begin()
        manager.put(setup, RowId("borrada"), b"x")
        manager.commit(setup)
        setup2 = manager.begin()
        manager.delete(setup2, RowId("borrada"))
        manager.commit(setup2)

        txn = manager.begin(IsolationLevel.SERIALIZABLE)
        assert manager.scan(txn) == []
        assert RowId("borrada") in manager._transactions[txn].read_set
