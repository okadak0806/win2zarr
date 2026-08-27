"""Convert WIN/WIN32 seismic waveform files to analysis-ready Zarr."""

from __future__ import annotations

from win2zarr._api import (
    ChannelSecond,
    ChannelTable,
    WinFile,
    WinSecond,
    WinTime,
    decode_win,
    decode_win_bytes,
    encode_win_bytes,
    open_zarr,
    write_zarr,
)
from win2zarr.channel_table import parse_channel_table

__all__ = [
    "ChannelSecond",
    "ChannelTable",
    "WinFile",
    "WinSecond",
    "WinTime",
    "decode_win",
    "decode_win_bytes",
    "encode_win_bytes",
    "open_zarr",
    "parse_channel_table",
    "write_zarr",
    "__version__",
]

try:
    from win2zarr._native import __version__ as __version__
except ImportError:  # pragma: no cover - source tree without maturin develop
    __version__ = "0.1.0"
