from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from win2zarr import ChannelSecond, WinFile, WinSecond, WinTime, encode_win_bytes


def test_cli_help():
    proc = subprocess.run(
        [sys.executable, "-m", "win2zarr.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "win-to-zarr" in proc.stdout
    assert "plan" in proc.stdout
    assert "convert" in proc.stdout


def test_cli_plan_and_convert(tmp_path):
    pytest.importorskip("zarr")
    samples = np.arange(100, dtype=np.int32) % 6
    raw = encode_win_bytes(
        WinFile(
            format="win",
            seconds=[
                WinSecond(
                    time=WinTime(2020, 1, 1, 0, 0, 0),
                    channels=[
                        ChannelSecond(0x1, 100, samples),
                        ChannelSecond(0x2, 100, samples),
                        ChannelSecond(0x3, 100, samples),
                    ],
                )
            ],
        )
    )
    win_path = tmp_path / "synth.win"
    win_path.write_bytes(raw)
    ch_path = tmp_path / "s.ch"
    ch_path.write_text(
        "0001 1 0 STA U 0 0 1 count 1 0 0 1 0 0 0\n"
        "0002 1 0 STA N 0 0 1 count 1 0 0 1 0 0 0\n"
        "0003 1 0 STA E 0 0 1 count 1 0 0 1 0 0 0\n",
        encoding="utf-8",
    )
    out = tmp_path / "zarr"

    plan = subprocess.run(
        [
            sys.executable,
            "-m",
            "win2zarr.cli",
            "plan",
            str(win_path),
            "--ch",
            str(ch_path),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert plan.returncode == 0, plan.stderr
    assert "20200101T0000.zarr" in plan.stdout
    assert not out.exists()

    convert = subprocess.run(
        [
            sys.executable,
            "-m",
            "win2zarr.cli",
            "convert",
            str(win_path),
            "--ch",
            str(ch_path),
            "--out",
            str(out),
            "--yes",
            "--counts",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert convert.returncode == 0, convert.stderr
    shard = out / "20200101T0000.zarr"
    assert shard.is_dir()

    dry = subprocess.run(
        [
            sys.executable,
            "-m",
            "win2zarr.cli",
            "convert",
            str(win_path),
            "--out",
            str(tmp_path / "unused"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert dry.returncode == 0, dry.stderr
    assert not (tmp_path / "unused").exists()
