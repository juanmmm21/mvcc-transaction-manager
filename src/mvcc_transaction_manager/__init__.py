"""mvcc-transaction-manager: control de concurrencia multiversión (MVCC) desde cero.

Mantiene snapshots consistentes por transacción sobre un storage subyacente
pluggable (ver `protocols.RowStore`) y soporta los tres niveles clásicos de
aislamiento SQL (read committed, repeatable read, serializable) sin que las
lecturas bloqueen nunca a las escrituras concurrentes.
"""

from __future__ import annotations

__version__ = "0.1.0"
