from __future__ import annotations

import numpy as np
import pytest

from win2zarr import (
    ChannelSecond,
    WinFile,
    WinSecond,
    WinTime,
    decode_win_bytes,
    encode_win_bytes,
    open_zarr,
    write_zarr,
)
from win2zarr.bf16 import bf16_bits_to_f32, f32_to_bf16_bits
from win2zarr.channel_table import parse_channel_table_text


def _minute_file(channels, nsec=3):
    seconds = []
    for s in range(nsec):
        chans = []
        for cid, data in channels:
            chans.append(ChannelSecond(cid, 100, np.asarray(data[s], dtype=np.int32)))
        seconds.append(
            WinSecond(time=WinTime(2020, 6, 15, 12, 30, s), channels=chans)
        )
    return WinFile(format="win", seconds=seconds)


def test_bf16_roundtrip_small_ints():
    x = np.array([0.0, 1.0, -2.0, 3.5, 100.0], dtype=np.float32)
    back = bf16_bits_to_f32(f32_to_bf16_bits(x))
    # bfloat16 keeps the top 7 mantissa bits; these values are exact or close.
    np.testing.assert_allclose(back, x, rtol=0, atol=0.015625)


def test_zarr_minute_shard_roundtrip(tmp_path):
    zarr = pytest.importorskip("zarr")
    _ = zarr
    u = [np.arange(100, dtype=np.int32) + s for s in range(3)]
    n = [np.full(100, s, dtype=np.int32) for s in range(3)]
    e = [-(np.arange(100, dtype=np.int32)) for s in range(3)]
    win = _minute_file([(0x0A01, u), (0x0A02, n), (0x0A03, e)])
    table = parse_channel_table_text(
        "0A01 1 0 N.ABC U 7 24 1 count 1 0.7 0 1 0 0 0\n"
        "0A02 1 0 N.ABC N 7 24 1 count 1 0.7 0 1 0 0 0\n"
        "0A03 1 0 N.ABC E 7 24 1 count 1 0.7 0 1 0 0 0\n"
    )
    paths = write_zarr(win, tmp_path, channel_table=table, physical=False)
    assert len(paths) == 1
    assert paths[0].name == "20200615T1230.zarr"

    shard = open_zarr(paths[0])
    assert shard.attrs["sample_rate"] == 100.0
    assert shard.attrs["timezone"] == "Asia/Tokyo"
    assert shard.n_samples == 6000
    assert shard.n_channels == 3
    assert shard.stations == ["N.ABC", "N.ABC", "N.ABC"]
    assert shard.components == ["U", "N", "E"]
    # First three seconds are valid; the rest of the minute is padded.
    assert shard.valid_mask[:300].all()
    assert not shard.valid_mask[300:].any()
    np.testing.assert_allclose(shard.waveforms[:100, 0], u[0], atol=1.0)
    np.testing.assert_allclose(shard.waveforms[100:200, 1], n[1], atol=1.0)


def test_encode_decode_then_zarr(tmp_path):
    pytest.importorskip("zarr")
    samples = ((np.arange(100, dtype=np.int32) % 5) - 2)
    original = WinFile(
        format="win",
        seconds=[
            WinSecond(
                time=WinTime(2021, 3, 4, 5, 6, 0),
                channels=[ChannelSecond(0x10, 100, samples)],
            )
        ],
    )
    decoded = decode_win_bytes(encode_win_bytes(original))
    paths = write_zarr(decoded, tmp_path, physical=False)
    shard = open_zarr(paths[0])
    np.testing.assert_allclose(shard.waveforms[:100, 0], samples, atol=1.0)
    assert shard.valid_mask[:100, 0].all()
    assert not shard.valid_mask[100:, 0].any()
