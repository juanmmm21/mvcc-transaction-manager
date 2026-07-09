"""Tests de propiedades: cargas aleatorias de miles de operaciones, con
semilla fija, comparadas contra un modelo de referencia independiente.

Dos propiedades distintas, cada una con su propio modelo de referencia:

1. Bajo una única transacción `READ_COMMITTED` activa a la vez (nunca dos
   solapadas), el estado final del motor debe coincidir exactamente con un
   `dict` de referencia actualizado en el mismo orden — y debe seguir
   coincidiendo después de intercalar llamadas a `gc()`.
2. Bajo muchas transacciones de los tres niveles de aislamiento abiertas y
   entrelazadas de forma aleatoria, cada lectura debe coincidir con lo que
   predice un log de commits (`row_id -> valor` por `commit_seq`) construido
   de forma independiente por el propio test, aplicando la regla de
   visibilidad documentada en `models.Snapshot` (`commit_seq <= horizonte`)
   — un modelo de referencia separado de la implementación, no una
   repetición de su código.
"""

from __future__ import annotations

import random

from mvcc_transaction_manager.models import CommitSeq, IsolationLevel, MvccError, RowId
from mvcc_transaction_manager.pipeline import MVCCTransactionManager
from mvcc_transaction_manager.store import InMemoryRowStore

_SEED = 20260709
_KEYS = [RowId(f"k{i}") for i in range(12)]


def test_serial_read_committed_workload_matches_reference_dict() -> None:
    rng = random.Random(_SEED)
    store = InMemoryRowStore()
    mgr = MVCCTransactionManager(store)
    reference: dict[RowId, bytes] = {}

    def assert_matches_reference() -> None:
        check = mgr.begin()
        expected = {k: v for k, v in reference.items()}
        actual = dict(mgr.scan(check))
        mgr.commit(check)
        assert actual == expected

    for iteration in range(4000):
        row_id = rng.choice(_KEYS)
        txn = mgr.begin()
        if rng.random() < 0.2 and row_id in reference:
            mgr.delete(txn, row_id)
            reference.pop(row_id, None)
        else:
            value = str(rng.randint(0, 1_000_000)).encode()
            mgr.put(txn, row_id, value)
            reference[row_id] = value
        mgr.commit(txn)

        if iteration % 137 == 0:
            # un lector repeatable_read abierto durante el gc no debe perder su versión
            reader = mgr.begin(IsolationLevel.REPEATABLE_READ)
            snapshot_before_gc = dict(mgr.scan(reader))
            mgr.gc()
            snapshot_after_gc = dict(mgr.scan(reader))
            assert snapshot_before_gc == snapshot_after_gc
            mgr.commit(reader)
            assert_matches_reference()

    assert_matches_reference()
    mgr.gc()
    assert_matches_reference()

    # tras el gc final (sin transacciones activas), cada fila viva conserva
    # exactamente una versión: la más reciente.
    for row_id in reference:
        assert len(store.versions_of(row_id)) == 1


def test_interleaved_multi_isolation_workload_matches_commit_log_visibility() -> None:
    rng = random.Random(_SEED + 1)
    mgr = MVCCTransactionManager(InMemoryRowStore())
    commit_log: list[tuple[CommitSeq, dict[RowId, bytes | None]]] = []
    live_horizon = CommitSeq(0)

    def reference_value_at(row_id: RowId, horizon: CommitSeq) -> bytes | None:
        value: bytes | None = None
        for commit_seq, writes in commit_log:
            if commit_seq > horizon:
                break
            if row_id in writes:
                value = writes[row_id]
        return value

    levels = list(IsolationLevel)
    open_transactions: dict[int, tuple[int, CommitSeq, dict[RowId, bytes | None]]] = {}
    # open_transactions[txn_id] = (isolation_index, begin_horizon, local_write_buffer)

    for _ in range(6000):
        action = rng.random()

        if action < 0.25 or not open_transactions:
            level = rng.choice(levels)
            txn = mgr.begin(level)
            open_transactions[txn] = (levels.index(level), live_horizon, {})
            continue

        txn = rng.choice(list(open_transactions.keys()))
        level_index, begin_horizon, local_writes = open_transactions[txn]
        level = levels[level_index]

        if action < 0.55:
            row_id = rng.choice(_KEYS)
            horizon = live_horizon if level is IsolationLevel.READ_COMMITTED else begin_horizon
            expected = local_writes.get(row_id, reference_value_at(row_id, horizon))
            actual = mgr.get(txn, row_id)
            assert actual == expected, (
                f"lectura inconsistente: txn={txn} nivel={level.value} fila={row_id} "
                f"esperado={expected!r} obtenido={actual!r}"
            )
        elif action < 0.80:
            row_id = rng.choice(_KEYS)
            value = str(rng.randint(0, 1_000_000)).encode()
            mgr.put(txn, row_id, value)
            local_writes[row_id] = value
        elif action < 0.90:
            row_id = rng.choice(_KEYS)
            mgr.delete(txn, row_id)
            local_writes[row_id] = None
        elif action < 0.95:
            try:
                commit_seq = mgr.commit(txn)
                commit_log.append((commit_seq, dict(local_writes)))
                live_horizon = commit_seq
            except MvccError:
                pass  # conflicto esperado en repeatable_read/serializable: se trata como abort
            del open_transactions[txn]
        else:
            mgr.abort(txn)
            del open_transactions[txn]

    # cierre ordenado de las transacciones que quedasen abiertas al final
    for txn, (_level_index, _begin_horizon, local_writes) in list(open_transactions.items()):
        try:
            commit_seq = mgr.commit(txn)
            commit_log.append((commit_seq, dict(local_writes)))
        except MvccError:
            pass

    verify = mgr.begin()
    for row_id in _KEYS:
        assert mgr.get(verify, row_id) == reference_value_at(row_id, CommitSeq(10**9))
    mgr.commit(verify)
