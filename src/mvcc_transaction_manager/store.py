"""Implementación en memoria del `Protocol` `RowStore`.

Sirve tanto para los tests como para la CLI de demostración. No es la única
implementación posible: cualquier estructura que cumpla `RowStore` (por
ejemplo un adaptador sobre `bplus-tree-storage-engine` o `lsm-tree-engine`)
puede sustituirla sin tocar `pipeline.py` — esa integración real ocurre
siempre dentro de `nanosql`, nunca por import cruzado desde este repositorio.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from .models import RowId, RowVersion


class InMemoryRowStore:
    """Almacén de versiones en memoria, indexado por fila mediante un dict.

    Estrategia de sincronización: esta clase **no es segura para acceso
    concurrente por sí misma** — no toma ningún lock internamente. Toda la
    serialización la impone `pipeline.MVCCTransactionManager` mediante un
    único lock global que envuelve cualquier llamada a esta clase (ver su
    docstring). Separar el storage "tonto" del control de concurrencia deja
    claro dónde vive el único punto de sincronización del módulo, en vez de
    repartir locks entre dos capas distintas.
    """

    def __init__(self) -> None:
        self._chains: dict[RowId, list[RowVersion]] = {}

    def append_version(self, version: RowVersion) -> None:
        self._chains.setdefault(version.row_id, []).append(version)

    def versions_of(self, row_id: RowId) -> Sequence[RowVersion]:
        return tuple(self._chains.get(row_id, ()))

    def all_versions(self) -> Iterator[RowVersion]:
        # Implementa el protocolo opcional `BulkRowStore`. Copia de la lista
        # de cadenas: el consumidor puede tardar en agotar el iterador y un
        # dict no admite mutación durante la iteración.
        for chain in list(self._chains.values()):
            yield from chain

    def row_ids(self) -> Iterator[RowId]:
        return iter(list(self._chains.keys()))

    def prune_versions(self, row_id: RowId, keep: Sequence[RowVersion]) -> int:
        existing = self._chains.get(row_id, [])
        removed = len(existing) - len(keep)
        if not keep:
            self._chains.pop(row_id, None)
        else:
            self._chains[row_id] = list(keep)
        return max(removed, 0)
