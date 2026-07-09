"""mvcc-transaction-manager: control de concurrencia multiversión (MVCC) desde cero.

Mantiene snapshots consistentes por transacción sobre un storage subyacente
pluggable (ver `protocols.RowStore`) y soporta los tres niveles clásicos de
aislamiento SQL (read committed, repeatable read, serializable) sin que las
lecturas bloqueen nunca a las escrituras concurrentes.
"""

from __future__ import annotations

from .models import (
    CommitSeq,
    IsolationLevel,
    MvccError,
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
from .pipeline import MVCCTransactionManager
from .protocols import RowStore
from .store import InMemoryRowStore

__version__ = "0.1.0"

__all__ = [
    "CommitSeq",
    "InMemoryRowStore",
    "IsolationLevel",
    "MVCCTransactionManager",
    "MvccError",
    "RowId",
    "RowStore",
    "RowVersion",
    "SerializationConflictError",
    "Snapshot",
    "Transaction",
    "TransactionId",
    "TransactionNotActiveError",
    "TransactionStatus",
    "WriteConflictError",
]
