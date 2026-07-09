"""Núcleo del control de concurrencia multiversión.

`MVCCTransactionManager` implementa los tres niveles de aislamiento sobre
un `RowStore` pluggable:

- Las escrituras de una transacción se acumulan en un buffer local
  (`Transaction.write_buffer`) y sólo se materializan en el `RowStore`
  dentro de la sección crítica de un `commit` exitoso. Esto es lo que
  permite que la regla de visibilidad de `models.Snapshot` sea una simple
  comparación de `commit_seq`: ninguna transacción ve nunca una versión a
  medio confirmar de otra, así que no hace falta un `xmin`/`xmax` por
  versión ni una tabla de estado de commits.
- `READ_COMMITTED` no fija un snapshot: cada lectura usa el horizonte de
  commit *actual* en el momento de la llamada.
- `REPEATABLE_READ` fija su snapshot en `begin` y lo reutiliza durante
  toda la transacción (snapshot isolation al estilo Postgres).
- `SERIALIZABLE` añade sobre `REPEATABLE_READ` detección de conflictos de
  serialización (write skew) mediante un algoritmo simplificado inspirado
  en SSI (Cahill et al.): se abortan las transacciones "pivote" que tienen
  a la vez una rw-antidependencia entrante y saliente con transacciones
  concurrentes.

Estrategia de sincronización: todo el estado compartido (tabla de
transacciones, contadores monotónicos, el `RowStore`) está protegido por un
único `threading.Lock` que envuelve cada operación pública. No se usa
locking de grano fino por fila porque el control de concurrencia de este
módulo es puramente optimista (MVCC), no pesimista — serializar el acceso
a filas individuales con locks bloqueantes es responsabilidad del
subproyecto hermano `lock-manager-deadlock-detector`, no de este.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from .models import (
    CommitSeq,
    IsolationLevel,
    RowId,
    RowVersion,
    Snapshot,
    Transaction,
    TransactionId,
    TransactionNotActiveError,
    TransactionStatus,
)
from .protocols import RowStore


class MVCCTransactionManager:
    """Gestor de transacciones MVCC sobre un `RowStore` pluggable."""

    def __init__(self, store: RowStore) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._next_txn_id = 1
        self._last_committed_seq = 0
        self._transactions: dict[TransactionId, Transaction] = {}

    # ------------------------------------------------------------------
    # Ciclo de vida de la transacción
    # ------------------------------------------------------------------

    def begin(
        self, isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
    ) -> TransactionId:
        """Registra una nueva transacción activa y le asigna su snapshot inicial.

        Para `READ_COMMITTED` este snapshot inicial es sólo un valor por
        defecto sin uso real: cada lectura recalcula su propio horizonte
        (ver `_read_horizon`).
        """
        with self._lock:
            txn_id = TransactionId(self._next_txn_id)
            self._next_txn_id += 1
            snapshot = Snapshot(holder=txn_id, commit_horizon=CommitSeq(self._last_committed_seq))
            self._transactions[txn_id] = Transaction(
                id=txn_id,
                isolation_level=isolation_level,
                status=TransactionStatus.ACTIVE,
                snapshot=snapshot,
            )
            return txn_id

    def status_of(self, txn_id: TransactionId) -> TransactionStatus:
        """Estado actual de una transacción conocida por el gestor."""
        with self._lock:
            txn = self._transactions.get(txn_id)
            if txn is None:
                raise TransactionNotActiveError(f"la transacción {txn_id} no existe")
            return txn.status

    # ------------------------------------------------------------------
    # Lecturas y escrituras
    # ------------------------------------------------------------------

    def get(self, txn_id: TransactionId, row_id: RowId) -> bytes | None:
        """Lee el valor visible de `row_id` para la transacción `txn_id`.

        Comprueba primero el buffer de escritura local (read-your-own-writes)
        antes de consultar el storage compartido con el horizonte de
        visibilidad correspondiente al nivel de aislamiento.
        """
        with self._lock:
            txn = self._require_active(txn_id)
            txn.read_set.add(row_id)
            if row_id in txn.write_buffer:
                return txn.write_buffer[row_id]
            horizon = self._read_horizon(txn)
            return self._visible_value(row_id, horizon)

    def put(self, txn_id: TransactionId, row_id: RowId, value: bytes) -> None:
        """Escribe (inserta o actualiza) `row_id` dentro del buffer local."""
        with self._lock:
            txn = self._require_active(txn_id)
            txn.write_buffer[row_id] = value
            txn.write_set.add(row_id)

    def delete(self, txn_id: TransactionId, row_id: RowId) -> None:
        """Marca `row_id` para borrado (tombstone) dentro del buffer local."""
        with self._lock:
            txn = self._require_active(txn_id)
            txn.write_buffer[row_id] = None
            txn.write_set.add(row_id)

    def scan(
        self, txn_id: TransactionId, row_ids: Iterable[RowId] | None = None
    ) -> list[tuple[RowId, bytes]]:
        """Lee varias filas a la vez, en orden determinista por `row_id`.

        Si `row_ids` es `None`, recorre todas las filas conocidas por el
        storage más las que esta transacción ha escrito localmente (para
        que un `insert` seguido de `scan` sin commit vea su propia
        escritura). Se usa tanto para uso normal como para las pruebas de
        phantom read: un `scan` repetido con el mismo snapshot nunca ve
        filas insertadas por otra transacción después de que el snapshot
        se tomase.
        """
        with self._lock:
            txn = self._require_active(txn_id)
            horizon = self._read_horizon(txn)
            candidates = (
                set(row_ids)
                if row_ids is not None
                else set(self._store.row_ids()) | set(txn.write_buffer)
            )
            results: list[tuple[RowId, bytes]] = []
            for row_id in sorted(candidates):
                txn.read_set.add(row_id)
                value = (
                    txn.write_buffer[row_id]
                    if row_id in txn.write_buffer
                    else self._visible_value(row_id, horizon)
                )
                if value is not None:
                    results.append((row_id, value))
            return results

    # ------------------------------------------------------------------
    # Confirmación y aborto (fase inicial, sin detección de conflictos aún)
    # ------------------------------------------------------------------

    def commit(self, txn_id: TransactionId) -> CommitSeq:
        """Confirma una transacción, materializando su buffer en el storage."""
        with self._lock:
            txn = self._require_active(txn_id)
            new_seq = CommitSeq(self._last_committed_seq + 1)
            for row_id, value in txn.write_buffer.items():
                self._store.append_version(
                    RowVersion(row_id=row_id, value=value, created_by=txn.id, commit_seq=new_seq)
                )
            txn.commit_seq = new_seq
            txn.status = TransactionStatus.COMMITTED
            self._last_committed_seq = new_seq
            return new_seq

    def abort(self, txn_id: TransactionId) -> None:
        """Aborta una transacción: descarta su buffer sin tocar el storage."""
        with self._lock:
            txn = self._require_active(txn_id)
            self._finalize_abort(txn)

    def _finalize_abort(self, txn: Transaction) -> None:
        txn.status = TransactionStatus.ABORTED
        txn.write_buffer.clear()
        del self._transactions[txn.id]

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _require_active(self, txn_id: TransactionId) -> Transaction:
        txn = self._transactions.get(txn_id)
        if txn is None or txn.status is not TransactionStatus.ACTIVE:
            raise TransactionNotActiveError(f"la transacción {txn_id} no existe o ya ha terminado")
        return txn

    def _read_horizon(self, txn: Transaction) -> CommitSeq:
        if txn.isolation_level is IsolationLevel.READ_COMMITTED:
            return CommitSeq(self._last_committed_seq)
        return txn.snapshot.commit_horizon

    def _visible_value(self, row_id: RowId, horizon: CommitSeq) -> bytes | None:
        visible = [v for v in self._store.versions_of(row_id) if v.commit_seq <= horizon]
        if not visible:
            return None
        newest = max(visible, key=lambda v: v.commit_seq)
        return newest.value
