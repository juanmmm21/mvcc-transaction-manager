"""CLI de demostración para mvcc-transaction-manager.

Subcomandos:

    demo    ejecuta un guion fijo que reproduce, sobre un `MVCCTransactionManager`
            en memoria, las tres anomalías clásicas de concurrencia (dirty read,
            non-repeatable read, phantom read) en cada nivel de aislamiento y el
            aborto por write skew en modo serializable.
    repl        intérprete interactivo de una sola sesión: permite abrir varias
                transacciones a la vez (identificadas por su id numérico) y
                entrelazar manualmente sus operaciones para explorar el
                comportamiento de cada nivel de aislamiento.
    benchmark   mide throughput real bajo contención: varios hilos compiten por
                incrementar filas de un keyspace pequeño, reintentando en el
                cliente ante cualquier conflicto, para cada nivel de aislamiento.

El `repl` es deliberadamente de un solo proceso y en memoria (no persiste
entre invocaciones): a diferencia de un motor de storage con estado en
disco, lo que este módulo demuestra es el entrelazado de transacciones
concurrentes, algo que sólo tiene sentido observar dentro de una misma
sesión.
"""

from __future__ import annotations

import argparse
import random
import shlex
import sys
import threading
import time

from mvcc_transaction_manager.models import (
    IsolationLevel,
    MvccError,
    RowId,
    TransactionId,
)
from mvcc_transaction_manager.pipeline import MVCCTransactionManager
from mvcc_transaction_manager.store import InMemoryRowStore

_LEVELS_BY_NAME = {level.value: level for level in IsolationLevel}


def _format_value(value: bytes | None) -> str:
    return "<none>" if value is None else value.decode(errors="replace")


def _run_demo() -> int:
    print("=== dirty read: ninguna transacción ve datos sin confirmar ===")
    for level in IsolationLevel:
        mgr = MVCCTransactionManager(InMemoryRowStore())
        setup = mgr.begin()
        mgr.put(setup, RowId("row"), b"committed")
        mgr.commit(setup)

        writer = mgr.begin()
        mgr.put(writer, RowId("row"), b"uncommitted")  # todavía no confirmada

        reader = mgr.begin(level)
        seen = mgr.get(reader, RowId("row"))
        print(f"  {level.value:16s} lee: {_format_value(seen)} (esperado: committed)")
        mgr.abort(writer)
        mgr.commit(reader)

    print()
    print("=== non-repeatable read: dos lecturas de la misma fila en una transacción ===")
    for level in IsolationLevel:
        mgr = MVCCTransactionManager(InMemoryRowStore())
        setup = mgr.begin()
        mgr.put(setup, RowId("row"), b"A")
        mgr.commit(setup)

        reader = mgr.begin(level)
        first = mgr.get(reader, RowId("row"))

        other = mgr.begin()
        mgr.put(other, RowId("row"), b"B")
        mgr.commit(other)

        second = mgr.get(reader, RowId("row"))
        mgr.commit(reader)
        changed = first != second
        print(
            f"  {level.value:16s} primera={_format_value(first)} "
            f"segunda={_format_value(second)} cambió={changed}"
        )

    print()
    print("=== phantom read: un scan repetido tras un insert concurrente ===")
    for level in IsolationLevel:
        mgr = MVCCTransactionManager(InMemoryRowStore())
        setup = mgr.begin()
        mgr.put(setup, RowId("row-1"), b"x")
        mgr.commit(setup)

        reader = mgr.begin(level)
        first_scan = mgr.scan(reader)

        other = mgr.begin()
        mgr.put(other, RowId("row-2"), b"y")
        mgr.commit(other)

        second_scan = mgr.scan(reader)
        mgr.commit(reader)
        phantom_appeared = len(second_scan) != len(first_scan)
        print(
            f"  {level.value:16s} filas antes={len(first_scan)} "
            f"después={len(second_scan)} phantom={phantom_appeared}"
        )

    print()
    print("=== write skew: dos médicos de guardia, modo serializable ===")
    mgr = MVCCTransactionManager(InMemoryRowStore())
    setup = mgr.begin()
    mgr.put(setup, RowId("alice"), b"on_call")
    mgr.put(setup, RowId("bob"), b"on_call")
    mgr.commit(setup)

    t1 = mgr.begin(IsolationLevel.SERIALIZABLE)
    t2 = mgr.begin(IsolationLevel.SERIALIZABLE)
    mgr.get(t1, RowId("alice"))
    mgr.get(t1, RowId("bob"))
    mgr.get(t2, RowId("alice"))
    mgr.get(t2, RowId("bob"))
    mgr.put(t1, RowId("alice"), b"off_call")
    mgr.put(t2, RowId("bob"), b"off_call")
    mgr.commit(t1)
    print("  t1 (pone a alice off_call) confirma correctamente")
    try:
        mgr.commit(t2)
        print("  t2 (pone a bob off_call) confirma -- ¡esto no debería pasar!")
        return 1
    except MvccError as exc:
        print(f"  t2 (pone a bob off_call) aborta como se espera: {exc}")
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    return _run_demo()


class _ReplSession:
    """Estado de una sesión interactiva: un único `MVCCTransactionManager`
    en memoria sobre el que se interpretan comandos de texto."""

    def __init__(self) -> None:
        self.manager = MVCCTransactionManager(InMemoryRowStore())

    def dispatch(self, line: str) -> bool:
        """Ejecuta un comando. Devuelve `False` si la sesión debe terminar."""
        parts = shlex.split(line)
        if not parts:
            return True
        command, *args = parts
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            print(f"comando desconocido: {command!r} (usa 'help')")
            return True
        try:
            return bool(handler(args))
        except MvccError as exc:
            print(f"error: {exc}")
            return True
        except (ValueError, IndexError) as exc:
            print(f"argumentos inválidos: {exc}")
            return True

    def _cmd_help(self, _args: list[str]) -> bool:
        print(
            "begin [read_committed|repeatable_read|serializable]\n"
            "get <txn> <row>\n"
            "put <txn> <row> <value>\n"
            "delete <txn> <row>\n"
            "scan <txn>\n"
            "commit <txn>\n"
            "abort <txn>\n"
            "gc\n"
            "status <txn>\n"
            "quit"
        )
        return True

    def _cmd_begin(self, args: list[str]) -> bool:
        level = _LEVELS_BY_NAME[args[0]] if args else IsolationLevel.READ_COMMITTED
        txn_id = self.manager.begin(level)
        print(f"txn={txn_id} isolation={level.value}")
        return True

    def _cmd_get(self, args: list[str]) -> bool:
        txn_id, row = TransactionId(int(args[0])), RowId(args[1])
        value = self.manager.get(txn_id, row)
        print(_format_value(value))
        return True

    def _cmd_put(self, args: list[str]) -> bool:
        txn_id, row, value = TransactionId(int(args[0])), RowId(args[1]), args[2]
        self.manager.put(txn_id, row, value.encode())
        return True

    def _cmd_delete(self, args: list[str]) -> bool:
        txn_id, row = TransactionId(int(args[0])), RowId(args[1])
        self.manager.delete(txn_id, row)
        return True

    def _cmd_scan(self, args: list[str]) -> bool:
        txn_id = TransactionId(int(args[0]))
        for row, value in self.manager.scan(txn_id):
            print(f"{row}={_format_value(value)}")
        return True

    def _cmd_commit(self, args: list[str]) -> bool:
        txn_id = TransactionId(int(args[0]))
        commit_seq = self.manager.commit(txn_id)
        print(f"committed commit_seq={commit_seq}")
        return True

    def _cmd_abort(self, args: list[str]) -> bool:
        txn_id = TransactionId(int(args[0]))
        self.manager.abort(txn_id)
        print("aborted")
        return True

    def _cmd_status(self, args: list[str]) -> bool:
        txn_id = TransactionId(int(args[0]))
        print(self.manager.status_of(txn_id).value)
        return True

    def _cmd_gc(self, _args: list[str]) -> bool:
        removed = self.manager.gc()
        print(f"pruned={removed}")
        return True

    def _cmd_quit(self, _args: list[str]) -> bool:
        return False

    _cmd_exit = _cmd_quit


def _benchmark_one_level(
    level: IsolationLevel, num_threads: int, ops_per_thread: int, num_keys: int
) -> None:
    """Mide throughput sostenido con un hilo de `gc()` periódico en segundo
    plano, como haría cualquier proceso de larga duración: sin él, la tabla
    de estado de conflicto (`_transactions`) crece sin límite a lo largo de
    la ejecución y la comprobación de `SERIALIZABLE` — que recorre todas las
    transacciones confirmadas retenidas — degrada a O(n) por commit. Esto se
    descubrió precisamente al correr este benchmark sin poda periódica; el
    número que se reporta aquí ya refleja el throughput sostenible.
    """
    mgr = MVCCTransactionManager(InMemoryRowStore())
    setup = mgr.begin()
    for i in range(num_keys):
        mgr.put(setup, RowId(f"key-{i}"), b"0")
    mgr.commit(setup)

    conflicts = 0
    conflicts_lock = threading.Lock()
    start_barrier = threading.Barrier(num_threads)
    stop_gc = threading.Event()

    def worker() -> None:
        nonlocal conflicts
        rng = random.Random()
        start_barrier.wait()
        for _ in range(ops_per_thread):
            while True:
                txn = mgr.begin(level)
                row = RowId(f"key-{rng.randrange(num_keys)}")
                current = mgr.get(txn, row)
                next_value = int(current) + 1 if current is not None else 1
                mgr.put(txn, row, str(next_value).encode())
                try:
                    mgr.commit(txn)
                    break
                except MvccError:
                    with conflicts_lock:
                        conflicts += 1

    def gc_loop() -> None:
        while not stop_gc.wait(timeout=0.005):
            mgr.gc()

    gc_thread = threading.Thread(target=gc_loop, daemon=True)
    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    started_at = time.perf_counter()
    gc_thread.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop_gc.set()
    gc_thread.join()
    elapsed = time.perf_counter() - started_at

    total_ops = num_threads * ops_per_thread
    throughput = total_ops / elapsed if elapsed > 0 else float("inf")
    retry_rate = conflicts / total_ops if total_ops else 0.0
    print(
        f"  {level.value:16s} committed={total_ops:5d} elapsed={elapsed:6.3f}s "
        f"throughput={throughput:8.0f} tx/s conflicts={conflicts:4d} retry_rate={retry_rate:6.2%}"
    )


def _cmd_benchmark(args: argparse.Namespace) -> int:
    print(
        f"=== throughput bajo contención: {args.threads} hilos x {args.ops_per_thread} "
        f"incrementos sobre {args.keys} filas compartidas ==="
    )
    for level in IsolationLevel:
        _benchmark_one_level(level, args.threads, args.ops_per_thread, args.keys)
    return 0


def _cmd_repl(_: argparse.Namespace) -> int:
    session = _ReplSession()
    print("mvcc-transaction-manager repl -- 'help' para ver los comandos, 'quit' para salir")
    for line in sys.stdin:
        if not session.dispatch(line.strip()):
            break
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvcc-transaction-manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="reproduce las anomalías clásicas por nivel de aislamiento")
    subparsers.add_parser("repl", help="intérprete interactivo multi-transacción")

    benchmark = subparsers.add_parser(
        "benchmark", help="mide throughput bajo contención por nivel de aislamiento"
    )
    benchmark.add_argument("--threads", type=int, default=8)
    benchmark.add_argument("--ops-per-thread", type=int, default=500)
    benchmark.add_argument("--keys", type=int, default=8)

    return parser


_HANDLERS = {
    "demo": _cmd_demo,
    "repl": _cmd_repl,
    "benchmark": _cmd_benchmark,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.command]
    try:
        return handler(args)
    except MvccError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
