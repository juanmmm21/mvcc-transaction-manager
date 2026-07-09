"""Detección de conflictos de escritura-escritura por write skew en modo
`SERIALIZABLE`, y su contraste deliberado con `REPEATABLE_READ`.

El escenario de referencia es el clásico de los dos médicos de guardia: la
política del hospital exige que al menos uno de los dos siga de guardia en
todo momento. Cada médico, antes de darse de baja, comprueba si el otro
sigue de guardia. Si ambas transacciones leen el estado "ambos de guardia"
y cada una decide darse de baja confiando en que el otro cubre, el
resultado final viola la política aunque ninguna escritura individual
pise a la otra (no hay conflicto de escritura-escritura clásico sobre la
misma fila) — por eso hace falta rastrear también las lecturas.
"""

from __future__ import annotations

import pytest

from mvcc_transaction_manager.models import (
    IsolationLevel,
    MvccError,
    RowId,
    SerializationConflictError,
)
from mvcc_transaction_manager.pipeline import MVCCTransactionManager
from mvcc_transaction_manager.store import InMemoryRowStore


def _hospital_with_two_doctors_on_call() -> MVCCTransactionManager:
    mgr = MVCCTransactionManager(InMemoryRowStore())
    setup = mgr.begin()
    mgr.put(setup, RowId("alice"), b"on_call")
    mgr.put(setup, RowId("bob"), b"on_call")
    mgr.commit(setup)
    return mgr


def _run_write_skew_scenario(mgr: MVCCTransactionManager, level: IsolationLevel) -> None:
    t1 = mgr.begin(level)
    t2 = mgr.begin(level)

    # cada transacción comprueba que el otro médico sigue de guardia antes de darse de baja
    assert mgr.get(t1, RowId("bob")) == b"on_call"
    assert mgr.get(t2, RowId("alice")) == b"on_call"

    mgr.put(t1, RowId("alice"), b"off_call")
    mgr.put(t2, RowId("bob"), b"off_call")

    mgr.commit(t1)
    mgr.commit(t2)


def test_serializable_detects_and_aborts_write_skew() -> None:
    mgr = _hospital_with_two_doctors_on_call()
    with pytest.raises(SerializationConflictError):
        _run_write_skew_scenario(mgr, IsolationLevel.SERIALIZABLE)


def test_repeatable_read_does_not_catch_write_skew() -> None:
    """Documenta la garantía real de `REPEATABLE_READ`: al no rastrear
    conjuntos de lectura para detección de conflictos, dos transacciones
    con write sets disjuntos pueden confirmar ambas aunque violen un
    invariante que depende de sus lecturas — exactamente la anomalía que
    `SERIALIZABLE` sí evita."""
    mgr = _hospital_with_two_doctors_on_call()
    _run_write_skew_scenario(mgr, IsolationLevel.REPEATABLE_READ)  # no lanza

    verify = mgr.begin()
    assert mgr.get(verify, RowId("alice")) == b"off_call"
    assert mgr.get(verify, RowId("bob")) == b"off_call"
    mgr.commit(verify)


def test_write_skew_survivor_keeps_its_committed_change() -> None:
    mgr = _hospital_with_two_doctors_on_call()
    t1 = mgr.begin(IsolationLevel.SERIALIZABLE)
    t2 = mgr.begin(IsolationLevel.SERIALIZABLE)
    mgr.get(t1, RowId("bob"))
    mgr.get(t2, RowId("alice"))
    mgr.put(t1, RowId("alice"), b"off_call")
    mgr.put(t2, RowId("bob"), b"off_call")

    mgr.commit(t1)
    with pytest.raises(SerializationConflictError):
        mgr.commit(t2)

    verify = mgr.begin()
    assert mgr.get(verify, RowId("alice")) == b"off_call"
    assert mgr.get(verify, RowId("bob")) == b"on_call"  # t2 abortó: bob sigue de guardia
    mgr.commit(verify)


def test_serializable_permits_disjoint_transactions_without_read_overlap() -> None:
    """Dos transacciones serializables que ni leen ni escriben filas en
    común deben poder confirmar ambas sin falso positivo."""
    mgr = MVCCTransactionManager(InMemoryRowStore())
    t1 = mgr.begin(IsolationLevel.SERIALIZABLE)
    t2 = mgr.begin(IsolationLevel.SERIALIZABLE)
    mgr.put(t1, RowId("row-a"), b"1")
    mgr.put(t2, RowId("row-b"), b"2")
    mgr.commit(t1)
    mgr.commit(t2)  # no debe lanzar


def test_serializable_three_transaction_pivot_aborts_only_the_pivot() -> None:
    """Escenario de 3 transacciones con dos rw-antidependencias
    consecutivas: T1 lee X y escribe Y; T2 (el pivote) lee Y y escribe Z;
    T3 lee Z y escribe X. Si las tres confirman en orden T1, T2, T3, T2 es
    quien completa el ciclo peligroso (tiene entrante desde T1 y saliente
    hacia T3) y debe abortar; T1 y T3 no forman parte de ningún ciclo entre
    sí y ambas confirman."""
    mgr = MVCCTransactionManager(InMemoryRowStore())
    setup = mgr.begin()
    mgr.put(setup, RowId("x"), b"0")
    mgr.put(setup, RowId("y"), b"0")
    mgr.put(setup, RowId("z"), b"0")
    mgr.commit(setup)

    t1 = mgr.begin(IsolationLevel.SERIALIZABLE)
    t2 = mgr.begin(IsolationLevel.SERIALIZABLE)
    t3 = mgr.begin(IsolationLevel.SERIALIZABLE)

    mgr.get(t1, RowId("x"))
    mgr.get(t2, RowId("y"))
    mgr.get(t3, RowId("z"))

    mgr.put(t1, RowId("y"), b"1")
    mgr.put(t2, RowId("z"), b"1")
    mgr.put(t3, RowId("x"), b"1")

    mgr.commit(t1)  # sin conflicto entrante todavía: confirma
    # t2 tiene entrante desde t1 (leyó y, t1 escribió y) y saliente hacia t3 (escribe z, t3 leyó z)
    with pytest.raises(SerializationConflictError):
        mgr.commit(t2)
    mgr.commit(t3)  # su única antidependencia era con t2, que abortó: confirma


def test_mvcc_error_is_the_common_base_for_conflict_errors() -> None:
    mgr = _hospital_with_two_doctors_on_call()
    with pytest.raises(MvccError):
        _run_write_skew_scenario(mgr, IsolationLevel.SERIALIZABLE)
