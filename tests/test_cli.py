"""Tests de humo de la CLI de demostración (`__main__.py`)."""

from __future__ import annotations

import io

import pytest

from mvcc_transaction_manager.__main__ import main


def test_demo_command_reports_expected_anomaly_outcomes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo"]) == 0
    output = capsys.readouterr().out

    assert "read_committed   lee: committed" in output
    assert "repeatable_read  lee: committed" in output
    assert "serializable     lee: committed" in output

    assert "read_committed   primera=A segunda=B cambió=True" in output
    assert "repeatable_read  primera=A segunda=A cambió=False" in output
    assert "serializable     primera=A segunda=A cambió=False" in output

    assert "read_committed   filas antes=1 después=2 phantom=True" in output
    assert "repeatable_read  filas antes=1 después=1 phantom=False" in output
    assert "serializable     filas antes=1 después=1 phantom=False" in output

    assert "t1 (pone a alice off_call) confirma correctamente" in output
    assert "t2 (pone a bob off_call) aborta como se espera" in output


def test_repl_processes_a_scripted_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = "\n".join(
        [
            "begin",
            "put 1 x hello",
            "commit 1",
            "begin repeatable_read",
            "get 2 x",
            "scan 2",
            "status 2",
            "gc",
            "quit",
        ]
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(script))

    assert main(["repl"]) == 0
    output = capsys.readouterr().out

    assert "txn=1 isolation=read_committed" in output
    assert "committed commit_seq=1" in output
    assert "txn=2 isolation=repeatable_read" in output
    assert "hello" in output
    assert "x=hello" in output
    assert "active" in output
    assert "pruned=0" in output


def test_repl_reports_errors_without_crashing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = "\n".join(["get 999 x", "quit"])
    monkeypatch.setattr("sys.stdin", io.StringIO(script))

    assert main(["repl"]) == 0
    output = capsys.readouterr().out
    assert "error:" in output


def test_repl_help_lists_available_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("help\nquit\n"))

    assert main(["repl"]) == 0
    output = capsys.readouterr().out
    assert "begin [read_committed|repeatable_read|serializable]" in output
    assert "commit <txn>" in output


def test_benchmark_command_reports_throughput_per_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "--threads", "2", "--ops-per-thread", "5", "--keys", "3"]) == 0
    output = capsys.readouterr().out
    assert "read_committed" in output
    assert "repeatable_read" in output
    assert "serializable" in output
    assert "tx/s" in output


def test_unknown_command_exits_with_argparse_error() -> None:
    with pytest.raises(SystemExit):
        main(["not-a-real-command"])
