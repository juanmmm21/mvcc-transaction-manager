"""Tests con transacciones simuladas en hilos reales, para verificar que el
lock global de `MVCCTransactionManager` sirve realmente para dos cosas
distintas: (1) que el estado compartido no se corrompe bajo acceso
concurrente real (no simulado por interleaving manual de un solo hilo), y
(2) que los snapshots de transacciones `REPEATABLE_READ`/`SERIALIZABLE`
permanecen aislados de escrituras concurrentes de otros hilos.
"""

from __future__ import annotations

import threading
import time

from mvcc_transaction_manager.models import IsolationLevel, MvccError, RowId
from mvcc_transaction_manager.pipeline import MVCCTransactionManager
from mvcc_transaction_manager.store import InMemoryRowStore


def test_concurrent_begin_assigns_unique_transaction_ids() -> None:
    mgr = MVCCTransactionManager(InMemoryRowStore())
    num_threads = 64
    ids: list[int] = []
    lock = threading.Lock()
    start_barrier = threading.Barrier(num_threads)

    def worker() -> None:
        start_barrier.wait()
        txn = mgr.begin()
        with lock:
            ids.append(txn)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(ids) == num_threads
    assert len(set(ids)) == num_threads  # ningún id duplicado pese a la carrera


def test_concurrent_repeatable_read_snapshots_stay_isolated() -> None:
    mgr = MVCCTransactionManager(InMemoryRowStore())
    setup = mgr.begin()
    mgr.put(setup, RowId("counter"), b"0")
    mgr.commit(setup)

    num_readers = 8
    num_writers = 20
    start_barrier = threading.Barrier(num_readers + num_writers)
    observations: list[tuple[bytes | None, bytes | None]] = [(None, None)] * num_readers
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def reader_worker(index: int) -> None:
        try:
            start_barrier.wait()
            txn = mgr.begin(IsolationLevel.REPEATABLE_READ)
            first = mgr.get(txn, RowId("counter"))
            time.sleep(0.01)  # deja hueco a que los escritores confirmen mientras tanto
            second = mgr.get(txn, RowId("counter"))
            mgr.commit(txn)
            observations[index] = (first, second)
        except BaseException as exc:  # noqa: BLE001 - se reporta desde el hilo principal
            with errors_lock:
                errors.append(exc)

    def writer_worker(value: int) -> None:
        try:
            start_barrier.wait()
            txn = mgr.begin()  # read committed: nunca conflictúa por escritura
            mgr.put(txn, RowId("counter"), str(value).encode())
            mgr.commit(txn)
        except BaseException as exc:  # noqa: BLE001 - se reporta desde el hilo principal
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=reader_worker, args=(i,)) for i in range(num_readers)]
    threads += [
        threading.Thread(target=writer_worker, args=(v,)) for v in range(1, num_writers + 1)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    for first, second in observations:
        assert first is not None and second is not None
        assert first == second, "el snapshot de un lector cambió pese a ser REPEATABLE_READ"


def test_concurrent_serializable_increments_are_never_lost() -> None:
    """`SERIALIZABLE` con reintento en el cliente ante conflicto debe
    comportarse como una suma atómica: ningún incremento se pierde aunque
    muchos hilos compitan por la misma fila."""
    mgr = MVCCTransactionManager(InMemoryRowStore())
    setup = mgr.begin()
    mgr.put(setup, RowId("counter"), b"0")
    mgr.commit(setup)

    num_threads = 12
    increments_per_thread = 8
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(increments_per_thread):
                while True:
                    txn = mgr.begin(IsolationLevel.SERIALIZABLE)
                    current = mgr.get(txn, RowId("counter"))
                    assert current is not None
                    mgr.put(txn, RowId("counter"), str(int(current) + 1).encode())
                    try:
                        mgr.commit(txn)
                        break
                    except MvccError:
                        continue  # otro hilo ganó la carrera: reintentar con snapshot fresco
        except BaseException as exc:  # noqa: BLE001 - se reporta desde el hilo principal
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    verify = mgr.begin()
    final_value = mgr.get(verify, RowId("counter"))
    mgr.commit(verify)
    assert final_value == str(num_threads * increments_per_thread).encode()
