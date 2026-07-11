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
from collections.abc import Iterable, Sequence

from .models import (
    CommitSeq,
    IsolationLevel,
    RowId,
    RowVersion,
    SerializationConflictError,
    Snapshot,
    Transaction,
    TransactionId,
    TransactionNotActiveError,
    TransactionStatus,
    WriteConflictError,
)
from .protocols import BulkRowStore, RowStore


class MVCCTransactionManager:
    """Gestor de transacciones MVCC sobre un `RowStore` pluggable."""

    def __init__(self, store: RowStore, *, last_committed_seq: int = 0) -> None:
        """`last_committed_seq` siembra el contador monotónico de commits.

        Necesario al reabrir un `RowStore` persistido de una sesión
        anterior: las versiones ya materializadas llevan sus `commit_seq`
        originales, y un contador que arrancase de nuevo en 0 las dejaría
        todas por encima de cualquier horizonte nuevo (nada sería visible)
        y reutilizaría seqs ya usados. El llamador pasa el `commit_seq`
        máximo presente en el storage; con un storage vacío, el 0 por
        defecto mantiene el comportamiento de siempre.
        """
        if last_committed_seq < 0:
            raise ValueError(f"last_committed_seq debe ser >= 0, recibido {last_committed_seq}")
        self._store = store
        self._lock = threading.Lock()
        self._next_txn_id = 1
        self._last_committed_seq = int(last_committed_seq)
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
            if row_ids is None and isinstance(self._store, BulkRowStore):
                return self._scan_bulk(self._store, txn, horizon)
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

    def _scan_bulk(
        self, store: BulkRowStore, txn: Transaction, horizon: CommitSeq
    ) -> list[tuple[RowId, bytes]]:
        """Camino rápido de `scan` para stores que implementan
        `BulkRowStore`: una única pasada sobre todas las versiones en vez de
        una llamada `versions_of` por fila. Semántica idéntica al camino
        fila a fila — misma visibilidad, mismo orden, mismo `read_set`
        (toda fila conocida cuenta como leída, visible o no: es lo que
        protege de phantom reads a la detección de write skew)."""
        newest: dict[RowId, RowVersion] = {}
        known: set[RowId] = set(txn.write_buffer)
        for version in store.all_versions():
            known.add(version.row_id)
            if version.commit_seq <= horizon:
                current = newest.get(version.row_id)
                if current is None or version.commit_seq > current.commit_seq:
                    newest[version.row_id] = version
        results: list[tuple[RowId, bytes]] = []
        for row_id in sorted(known):
            txn.read_set.add(row_id)
            if row_id in txn.write_buffer:
                value = txn.write_buffer[row_id]
            else:
                version_or_none = newest.get(row_id)
                value = version_or_none.value if version_or_none is not None else None
            if value is not None:
                results.append((row_id, value))
        return results

    # ------------------------------------------------------------------
    # Confirmación y aborto
    # ------------------------------------------------------------------

    def commit(self, txn_id: TransactionId) -> CommitSeq:
        """Confirma una transacción, materializando su buffer en el storage.

        En `REPEATABLE_READ` y `SERIALIZABLE` valida antes que ninguna otra
        transacción haya confirmado, después de que ésta tomase su
        snapshot, un cambio sobre alguna fila de su `write_set`
        ('first committer wins'); si lo hizo, aborta con
        `WriteConflictError` en vez de sobrescribir a ciegas datos que esta
        transacción nunca llegó a ver. `READ_COMMITTED` no hace esta
        comprobación: como cada lectura suya ya usa el estado confirmado
        más reciente, no hay una noción de "dato obsoleto" que proteger.

        En `SERIALIZABLE` además comprueba conflictos de serialización
        (write skew) vía `_check_serialization_conflicts` antes de dar la
        confirmación por buena.
        """
        with self._lock:
            txn = self._require_active(txn_id)
            try:
                if txn.isolation_level is not IsolationLevel.READ_COMMITTED:
                    self._check_write_write_conflicts(txn)
                if txn.isolation_level is IsolationLevel.SERIALIZABLE:
                    self._check_serialization_conflicts(txn)
            except (WriteConflictError, SerializationConflictError):
                self._finalize_abort(txn)
                raise

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

    def _check_write_write_conflicts(self, txn: Transaction) -> None:
        """Lanza `WriteConflictError` si `txn` pisaría un cambio que no vio.

        Para cada fila que `txn` quiere escribir, comprueba si la versión
        más reciente ya confirmada tiene un `commit_seq` posterior al
        horizonte del snapshot de `txn` — es decir, si alguien más
        confirmó un cambio sobre esa fila *después* de que `txn` empezase.
        Como el storage sólo contiene versiones ya confirmadas (nunca
        material de una transacción todavía activa), esta comprobación no
        necesita distinguir "confirmado" de "en curso": basta comparar
        `commit_seq` contra el horizonte.
        """
        for row_id in txn.write_set:
            chain = self._store.versions_of(row_id)
            if not chain:
                continue
            latest = max(chain, key=lambda v: v.commit_seq)
            if latest.created_by == txn.id:
                continue
            if latest.commit_seq > txn.snapshot.commit_horizon:
                raise WriteConflictError(
                    f"la transacción {txn.id} no puede confirmar: la fila {row_id!r} "
                    f"fue modificada por la transacción {latest.created_by} "
                    f"(commit_seq={latest.commit_seq}) después del snapshot de {txn.id} "
                    f"(commit_horizon={txn.snapshot.commit_horizon})"
                )

    def _check_serialization_conflicts(self, txn: Transaction) -> None:
        """Detecta y aborta transacciones "pivote" en modo `SERIALIZABLE`.

        Implementa una versión simplificada de la regla de estructura
        peligrosa de SSI (Cahill, Röhm y Fekete, 2008): si `txn` tiene, en
        el momento de confirmar, tanto una rw-antidependencia *entrante*
        (otra transacción concurrente ya confirmó una escritura sobre algo
        que `txn` leyó) como una *saliente* (`txn` está a punto de escribir
        algo que otra transacción concurrente ya leyó), entonces `txn` es
        el vértice central de un posible ciclo de serialización de 2 o 3
        transacciones y debe abortar — abortar al vértice central basta
        para romper cualquier ciclo que pasase por él, sin necesidad de
        reconstruir el grafo de dependencias completo.

        El caso clásico de write skew (dos transacciones que leen las
        mismas filas y escriben cada una en una fila distinta según lo
        leído) es un ciclo de 2 transacciones: cuando la segunda en
        confirmar hace su propia comprobación, su antidependencia entrante
        viene de la primera (ya confirmada) y su saliente se detecta
        porque la primera, aunque ya confirmada, sigue contando como
        concurrente (confirmó después de que la segunda tomase su
        snapshot) y había leído la fila que la segunda escribe.

        Nótese que aquí sólo se *lee* el estado de otras transacciones,
        nunca se muta: cada transacción calcula sus propios `has_conflict_in`/
        `has_conflict_out` exclusivamente a partir de transacciones que ya
        confirmaron (o siguen activas) en el momento de su propia
        comprobación. Marcar proactivamente el `has_conflict_in` de otra
        transacción todavía activa sería incorrecto — si esta transacción
        (`txn`) acaba abortando por otro motivo, esa marca quedaría
        huérfana en una transacción que nunca tuvo una antidependencia real
        confirmada.
        """
        # Entrante: ¿alguien que ya confirmó escribió algo que yo leí,
        # habiendo confirmado después de que yo tomase mi snapshot?
        for other in self._transactions.values():
            if other.id == txn.id or other.status is not TransactionStatus.COMMITTED:
                continue
            assert other.commit_seq is not None
            if other.commit_seq <= txn.snapshot.commit_horizon:
                continue  # confirmó antes de que yo empezase: no es concurrente
            if txn.read_set & other.write_set:
                txn.has_conflict_in = True

        # Saliente: ¿estoy a punto de escribir algo que otra transacción
        # concurrente (activa, o ya confirmada después de que yo empezase)
        # ya leyó?
        for other in self._transactions.values():
            if other.id == txn.id or other.status is TransactionStatus.ABORTED:
                continue
            is_concurrent = other.status is TransactionStatus.ACTIVE or (
                other.commit_seq is not None and other.commit_seq > txn.snapshot.commit_horizon
            )
            if not is_concurrent:
                continue
            if txn.write_set & other.read_set:
                txn.has_conflict_out = True

        if txn.has_conflict_in and txn.has_conflict_out:
            raise SerializationConflictError(
                f"la transacción {txn.id} forma un ciclo de dependencias "
                "read-write con otra transacción concurrente (write skew): "
                "tiene tanto una rw-antidependencia entrante como saliente"
            )

    def _finalize_abort(self, txn: Transaction) -> None:
        txn.status = TransactionStatus.ABORTED
        txn.write_buffer.clear()
        del self._transactions[txn.id]

    # ------------------------------------------------------------------
    # Recolección de basura
    # ------------------------------------------------------------------

    def gc(self) -> int:
        """Poda versiones y estado de conflicto que ninguna transacción
        activa puede ya necesitar.

        El horizonte de poda es el mínimo `commit_horizon` entre todos los
        snapshots activos (o `None`, tratado como "infinito", si no hay
        ninguna transacción activa). Para cada fila, se conserva la versión
        más reciente con `commit_seq <= horizonte` (la "versión suelo": es
        la que vería cualquier transacción activa con el horizonte mínimo)
        junto con todas las versiones posteriores al horizonte; el resto —
        estrictamente más antiguo que la versión suelo — es inalcanzable
        para siempre, así que se elimina. Si la versión suelo es además la
        más reciente de la cadena y es un tombstone, la fila entera puede
        olvidarse: nadie, ni ahora ni en el futuro, volverá a necesitar
        saber que existió.

        Por la misma razón (ningún snapshot activo tiene un horizonte
        anterior al mínimo), también se descarta de la tabla de
        transacciones el estado de conflicto de las transacciones ya
        confirmadas con `commit_seq` anterior al horizonte: ninguna
        transacción, activa ahora o futura, podrá considerarlas ya
        concurrentes en `_check_serialization_conflicts`.

        Devuelve el número de versiones de fila eliminadas.
        """
        with self._lock:
            active = [
                t for t in self._transactions.values() if t.status is TransactionStatus.ACTIVE
            ]
            horizon: CommitSeq | None = (
                CommitSeq(min(t.snapshot.commit_horizon for t in active)) if active else None
            )

            pruned = 0
            for row_id in list(self._store.row_ids()):
                chain = sorted(self._store.versions_of(row_id), key=lambda v: v.commit_seq)
                if not chain:
                    continue
                if horizon is None:
                    floor_index = len(chain) - 1
                else:
                    floor_index = -1
                    for index, version in enumerate(chain):
                        if version.commit_seq <= horizon:
                            floor_index = index
                        else:
                            break
                    if floor_index == -1:
                        continue  # todas las versiones son posteriores al horizonte: nada que podar
                floor_version = chain[floor_index]
                row_fully_obsolete = floor_version.value is None and floor_index == len(chain) - 1
                keep: Sequence[RowVersion] = () if row_fully_obsolete else chain[floor_index:]
                pruned += self._store.prune_versions(row_id, keep)

            if horizon is None:
                stale_txn_ids = [
                    tid
                    for tid, t in self._transactions.items()
                    if t.status is TransactionStatus.COMMITTED
                ]
            else:
                stale_txn_ids = [
                    tid
                    for tid, t in self._transactions.items()
                    if t.status is TransactionStatus.COMMITTED
                    and t.commit_seq is not None
                    and t.commit_seq < horizon
                ]
            for tid in stale_txn_ids:
                del self._transactions[tid]

            return pruned

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
