"""CLI: ``win-to-zarr plan`` / ``win-to-zarr convert``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _expand_inputs(inputs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            files.append(path)
    return files


def _load_files(paths: Sequence[Path]):
    from win2zarr._api import decode_win

    decoded = []
    errors = []
    for path in paths:
        try:
            decoded.append(decode_win(path))
        except Exception as exc:
            errors.append((path, exc))
    return decoded, errors


def _plan(decoded, sample_rate: float, channel_table, physical: bool):
    from win2zarr.zarr_io import plan_shards

    return plan_shards(
        decoded, sample_rate=sample_rate, channel_table=channel_table, physical=physical
    )


def _print_plan(shards, out_dir: Path, errors) -> None:
    n_ch = shards[0].n_channels if shards else 0
    n_samp = shards[0].n_samples if shards else 0
    print(f"minute shards : {len(shards)}")
    print(f"sample rate   : {shards[0].sample_rate if shards else '-'} Hz")
    print(f"shape         : ({n_samp}, {n_ch})  chunks (6000, 3)  dtype=bfloat16_bits")
    print(f"timezone      : Asia/Tokyo (WIN BCD time is JST)")
    print(f"output dir    : {out_dir}")
    if errors:
        print(f"decode errors : {len(errors)}")
        for path, exc in errors[:10]:
            print(f"  {path}: {exc}")
    for shard in shards:
        n_valid = int(shard.valid_mask.sum())
        print(
            f"  {shard.shard_name}  "
            f"channels={shard.n_channels}  "
            f"valid_samples={n_valid}/{shard.n_samples * max(shard.n_channels, 1)}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="win-to-zarr",
        description="Convert WIN/WIN32 waveform files to minute-sharded Zarr stores.",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    sub = parser.add_subparsers(dest="command")

    def add_io(p: argparse.ArgumentParser) -> None:
        p.add_argument("inputs", nargs="+", help="WIN/WIN32 files or directories")
        p.add_argument("--out", required=True, type=Path, help="Output directory")
        p.add_argument("--ch", type=Path, default=None, help="Channel table (.ch)")
        p.add_argument("--sample-rate", type=float, default=100.0, help="Hz (default 100)")
        p.add_argument(
            "--counts",
            action="store_true",
            help="Store integer counts as float instead of physical units",
        )
        p.add_argument("--yes", action="store_true", help="Do not prompt before writing")
        p.add_argument("--dry-run", action="store_true", help="Plan only; do not write")

    plan = sub.add_parser("plan", help="Show the minute-shard conversion plan")
    add_io(plan)
    convert = sub.add_parser("convert", help="Write Zarr minute shards")
    add_io(convert)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version or args.command is None:
        from win2zarr import __version__

        if args.version or args.command is None and argv is None and len(sys.argv) == 2 and sys.argv[1] in {
            "-h",
            "--help",
        }:
            parser.print_help()
            return 0
        if args.version:
            print(__version__)
            return 0
        if args.command is None:
            parser.print_help()
            return 0

    paths = _expand_inputs(args.inputs)
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"error: input not found: {missing[0]}", file=sys.stderr)
        return 2

    table = None
    if args.ch is not None:
        from win2zarr.channel_table import parse_channel_table

        table = parse_channel_table(args.ch)

    decoded, errors = _load_files(paths)
    if not decoded:
        print("error: no WIN files decoded", file=sys.stderr)
        for path, exc in errors:
            print(f"  {path}: {exc}", file=sys.stderr)
        return 1

    shards = _plan(decoded, args.sample_rate, table, physical=not args.counts)
    _print_plan(shards, args.out, errors)

    if args.command == "plan" or args.dry_run:
        return 0

    if not args.yes:
        reply = input("Write Zarr shards? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("aborted")
            return 1

    from win2zarr.zarr_io import write_minute_shards

    written = write_minute_shards(shards, args.out, overwrite=True)
    print(f"wrote {len(written)} shard(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
