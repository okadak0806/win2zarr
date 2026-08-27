"""Import alias for the historical ``win_loader`` package name."""

from win2zarr import (
    ChannelSecond,
    ChannelTable,
    WinFile,
    WinSecond,
    WinTime,
    __version__,
    decode_win,
    decode_win_bytes,
    encode_win_bytes,
    open_zarr,
    parse_channel_table,
    write_zarr,
)

__all__ = [
    "ChannelSecond",
    "ChannelTable",
    "WinFile",
    "WinSecond",
    "WinTime",
    "__version__",
    "decode_win",
    "decode_win_bytes",
    "encode_win_bytes",
    "open_zarr",
    "parse_channel_table",
    "write_zarr",
]
