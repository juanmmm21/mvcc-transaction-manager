"""Tipos de dominio del gestor de transacciones MVCC.

El diseño sigue el modelo de visibilidad de Postgres pero simplificado para
un motor educativo de referencia: en vez de un `xmin`/`xmax` por versión más
una lista de transacciones activas en el momento del snapshot, cada versión
de fila lleva únicamente el identificador de la transacción que la creó
(`created_by`) y un número de secuencia de commit monotónico (`commit_seq`).

Esta simplificación es posible porque las versiones sólo se materializan en
el storage subyacente dentro de la sección crítica de `commit` (ver
`pipeline.MVCCTransactionManager`): nunca existe una versión "a medio
confirmar" visible para otra transacción, así que no hace falta consultar
una tabla de estado de commits (el equivalente al `clog` de Postgres) para
decidir si una versión es visible — basta comparar `commit_seq` contra el
`commit_horizon` congelado en el snapshot del lector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NewType

RowId = NewType("RowId", str)
"""Identificador lógico de una fila dentro del `RowStore` subyacente."""

TransactionId = NewType("TransactionId", int)
"""Identificador de transacción, asignado de forma monotónica en `begin`."""

CommitSeq = NewType("CommitSeq", int)
"""Número de secuencia de commit, asignado de forma monotónica en `commit`.

Determina el orden total de las escrituras confirmadas, independiente del
orden en que las transacciones fueron creadas (`TransactionId`) — dos
transacciones pueden confirmar en un orden distinto al que empezaron."""


class IsolationLevel(Enum):
    """Los tres niveles clásicos de aislamiento SQL soportados."""

    READ_COMMITTED = "read_committed"
    """Cada lectura toma un snapshot nuevo: nunca hay dirty read, pero sí
    puede haber non-repeatable read y phantom read entre dos lecturas de la
    misma transacción."""

    REPEATABLE_READ = "repeatable_read"
    """Snapshot fijado en `begin` y reutilizado durante toda la transacción
    (snapshot isolation al estilo Postgres): ni dirty read, ni
    non-repeatable read, ni phantom read, porque el snapshot nunca cambia.
    Esto es más fuerte que el mínimo exigido por el estándar SQL para este
    nivel (que sólo prohíbe non-repeatable read), pero es el comportamiento
    real de `REPEATABLE READ` en Postgres y se documenta aquí como decisión
    de diseño deliberada."""

    SERIALIZABLE = "serializable"
    """Igual que `REPEATABLE_READ` más detección de conflictos de
    escritura-escritura entre transacciones concurrentes con dependencias
    read-write cruzadas (write skew), abortando explícitamente una de las
    dos transacciones implicadas antes de permitir que ambas confirmen."""


class TransactionStatus(Enum):
    """Estado del ciclo de vida de una transacción."""

    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class RowVersion:
    """Una versión inmutable de una fila, ya confirmada.

    `value is None` representa un tombstone (la fila fue borrada por
    `created_by`), siguiendo la misma convención que `lsm-tree-engine` para
    marcar borrados sin necesitar un campo booleano aparte.
    """

    row_id: RowId
    value: bytes | None
    created_by: TransactionId
    commit_seq: CommitSeq

    def __post_init__(self) -> None:
        if self.commit_seq < 1:
            raise ValueError(f"commit_seq debe ser >= 1, recibido {self.commit_seq}")


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Vista congelada del estado confirmado en el momento en que se toma.

    Toda versión con `commit_seq <= commit_horizon` es visible para el
    poseedor de este snapshot (además de sus propias escrituras no
    confirmadas, resueltas aparte vía el buffer de escritura local de la
    transacción — ver `Transaction.write_buffer`).
    """

    holder: TransactionId
    commit_horizon: CommitSeq


@dataclass(slots=True)
class Transaction:
    """Estado mutable de una transacción activa o ya terminada.

    Se conserva en memoria por `MVCCTransactionManager` incluso después de
    confirmar, hasta que `gc()` determina que ninguna transacción todavía
    activa puede necesitar sus conjuntos de lectura/escritura para detectar
    conflictos de serialización (ver docstring de `pipeline.gc`).
    """

    id: TransactionId
    isolation_level: IsolationLevel
    status: TransactionStatus
    snapshot: Snapshot
    write_buffer: dict[RowId, bytes | None] = field(default_factory=dict)
    """Escrituras locales aún no confirmadas: `None` representa un borrado
    pendiente. Se consulta antes que el storage compartido para que una
    transacción siempre vea sus propias escrituras (read-your-own-writes)."""
    read_set: set[RowId] = field(default_factory=set)
    write_set: set[RowId] = field(default_factory=set)
    commit_seq: CommitSeq | None = None
    has_conflict_in: bool = False
    """Marca si alguna transacción concurrente ya confirmó una escritura
    sobre una fila que esta transacción leyó (rw-antidependencia entrante).
    Sólo se usa en modo `SERIALIZABLE`."""
    has_conflict_out: bool = False
    """Marca si esta transacción escribió una fila que otra transacción
    concurrente (activa o ya confirmada) había leído (rw-antidependencia
    saliente). Sólo se usa en modo `SERIALIZABLE`."""


class MvccError(Exception):
    """Error base de todo el módulo."""


class TransactionNotActiveError(MvccError):
    """Se intentó leer, escribir, confirmar o abortar una transacción que
    no existe o que ya terminó (commit o abort previos)."""


class WriteConflictError(MvccError):
    """Conflicto escritura-escritura ('first committer wins'): otra
    transacción confirmó un cambio sobre una fila de `write_set` después de
    que esta transacción tomase su snapshot. Aplica en `REPEATABLE_READ` y
    `SERIALIZABLE`; `READ_COMMITTED` nunca lanza este error porque cada
    escritura se basa en el estado confirmado más reciente."""


class SerializationConflictError(MvccError):
    """Conflicto de serialización en modo `SERIALIZABLE`: esta transacción
    forma parte de un ciclo de dependencias read-write (ejemplo clásico:
    write skew) junto con otra transacción concurrente y debe abortar para
    preservar el equivalente a una ejecución serial."""
