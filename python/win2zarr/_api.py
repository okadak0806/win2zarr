from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from win2zarr.channel_table import ChannelRecord, ChannelTable, parse_channel_table
from win2zarr.zarr_io import MinuteShard, open_zarr_store, plan_shards, write_minute_shards

PathLike = Union[str, Path]
JST = timezone(timedelta(hours=9), name="Asia/Tokyo")


@dataclass(frozen=True, order=True)
class WinTime:
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    timezone: str = "Asia/Tokyo"

    def to_datetime(self) -> datetime:
        return datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            tzinfo=JST,
        )

    def minute_key(self) -> "WinTime":
        return WinTime(
            self.year, self.month, self.day, self.hour, self.minute, 0, self.timezone
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WinTime":
        return cls(
            year=int(data["year"]),
            month=int(data["month"]),
            day=int(data["day"]),
            hour=int(data["hour"]),
            minute=int(data["minute"]),
            second=int(data["second"]),
            timezone=str(data.get("timezone", "Asia/Tokyo")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "hour": self.hour,
            "minute": self.minute,
            "second": self.second,
            "timezone": self.timezone,
        }


@dataclass
class ChannelSecond:
    channel_id: int
    sample_rate: int
    samples: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": int(self.channel_id),
            "sample_rate": int(self.sample_rate),
            "samples": np.asarray(self.samples, dtype=np.int32),
        }


@dataclass
class WinSecond:
    time: WinTime
    channels: List[ChannelSecond] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time.to_dict(),
            "channels": [ch.to_dict() for ch in self.channels],
        }


@dataclass
class WinFile:
    format: str
    seconds: List[WinSecond]
    source: Optional[str] = None

    @property
    def channel_ids(self) -> List[int]:
        seen = []
        for sec in self.seconds:
            for ch in sec.channels:
                if ch.channel_id not in seen:
                    seen.append(ch.channel_id)
        return seen


def _require_native():
    try:
        from win2zarr import _native
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "win2zarr native extension is not built. "
            "Install with `pip install -e .` or `maturin develop`."
        ) from exc
    return _native


def _file_from_native(payload: Mapping[str, Any], source: Optional[str] = None) -> WinFile:
    seconds = []
    for sec in payload["seconds"]:
        channels = [
            ChannelSecond(
                channel_id=int(ch["channel_id"]),
                sample_rate=int(ch["sample_rate"]),
                samples=np.asarray(ch["samples"], dtype=np.int32),
            )
            for ch in sec["channels"]
        ]
        seconds.append(WinSecond(time=WinTime.from_mapping(sec["time"]), channels=channels))
    return WinFile(format=str(payload["format"]), seconds=seconds, source=source)


def decode_win_bytes(data: bytes) -> WinFile:
    """Decode a WIN or WIN32 buffer."""
    native = _require_native()
    return _file_from_native(native.decode_win_bytes(data))


def decode_win(path: PathLike) -> WinFile:
    """Decode a WIN or WIN32 file from disk."""
    path = Path(path)
    payload = decode_win_bytes(path.read_bytes())
    payload.source = str(path)
    return payload


def encode_win_bytes(win_file: Union[WinFile, Sequence[WinSecond]], format: str = "win") -> bytes:
    """Encode second blocks to on-disk WIN or WIN32 bytes."""
    native = _require_native()
    if isinstance(win_file, WinFile):
        format = win_file.format
        seconds = [sec.to_dict() for sec in win_file.seconds]
    else:
        seconds = [
            sec.to_dict() if isinstance(sec, WinSecond) else sec for sec in win_file
        ]
    return bytes(native.encode_win_bytes(format, seconds))


def write_zarr(
    win_file: Union[WinFile, Sequence[WinFile]],
    out_dir: PathLike,
    channel_table: Optional[Union[ChannelTable, PathLike]] = None,
    sample_rate: float = 100.0,
    physical: bool = True,
    overwrite: bool = True,
) -> List[Path]:
    """Write decoded WIN data as minute-sharded Zarr stores.

    Golden-path layout (100 Hz):
    ``<out_dir>/<YYYYMMDDTHHMM>.zarr`` with ``waveforms`` uint16 BF16 bits
    of shape ``(n_samples, n_ch)`` chunked ``(6000, 3)``, plus ``valid_mask``.
    """
    files = [win_file] if isinstance(win_file, WinFile) else list(win_file)
    table = None
    if channel_table is not None:
        table = (
            channel_table
            if isinstance(channel_table, ChannelTable)
            else parse_channel_table(channel_table)
        )
    shards = plan_shards(files, sample_rate=sample_rate, channel_table=table, physical=physical)
    return write_minute_shards(shards, out_dir, overwrite=overwrite)


def open_zarr(path: PathLike) -> MinuteShard:
    """Read a minute-shard Zarr store written by :func:`write_zarr`."""
    return open_zarr_store(path)


def load_channel_table(path: PathLike) -> ChannelTable:
    return parse_channel_table(path)
