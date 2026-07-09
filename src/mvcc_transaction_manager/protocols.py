"""Interfaz del storage subyacente sobre el que este módulo versiona filas.

`mvcc-transaction-manager` no importa código de `bplus-tree-storage-engine`
ni de `lsm-tree-engine` (la integración real ocurre siempre dentro de
`nanosql`, nunca por import cruzado entre repositorios del ecosistema); en
su lugar define este `Protocol` mínimo y lo implementa en memoria
(`store.InMemoryRowStore`) para poder probar el control de concurrencia de
forma aislada. Cualquier adaptador sobre un motor de storage real que
cumpla esta interfaz puede sustituir la implementación en memoria sin tocar
`pipeline.py`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from .models import RowId, RowVersion


@runtime_checkable
class RowStore(Protocol):
    """Almacén de versiones de fila, indexado por `RowId`.

    Es deliberadamente "tonto": no conoce transacciones, snapshots ni
    reglas de visibilidad — sólo guarda cadenas de versiones ya decididas
    (nunca una versión a medio confirmar) y las expone para que
    `pipeline.MVCCTransactionManager` aplique sobre ellas toda la lógica de
    control de concurrencia.
    """

    def append_version(self, version: RowVersion) -> None:
        """Añade una nueva versión, ya confirmada, a la cadena de su fila.

        El invariante que garantiza el llamador es que sólo se invoca
        dentro de la sección crítica de un `commit` exitoso: ninguna
        versión de una transacción abortada llega nunca a este método.
        """
        ...

    def versions_of(self, row_id: RowId) -> Sequence[RowVersion]:
        """Cadena completa de versiones de una fila, en cualquier orden.

        El llamador es responsable de ordenar por `commit_seq` si lo
        necesita; el propio storage no garantiza orden de inserción porque
        no todos los backends plugables pueden ofrecerlo barato.
        """
        ...

    def row_ids(self) -> Iterator[RowId]:
        """Todas las filas que tienen o han tenido al menos una versión."""
        ...

    def prune_versions(self, row_id: RowId, keep: Sequence[RowVersion]) -> int:
        """Sustituye la cadena de `row_id` por `keep` (calculada por el
        recolector de basura de `pipeline.gc`). Si `keep` está vacío, la
        fila desaparece por completo del storage. Devuelve cuántas
        versiones se eliminaron.
        """
        ...
