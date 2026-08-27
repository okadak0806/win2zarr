from __future__ import annotations

import numpy as np

from win2zarr import (
    ChannelSecond,
    WinFile,
    WinSecond,
    WinTime,
    decode_win_bytes,
    encode_win_bytes,
)


def _samples(n: int = 100, start: int = 0) -> np.ndarray:
    return ((np.arange(n, dtype=np.int32) + start) % 7) - 3


def _second(second: int, channels) -> WinSecond:
    return WinSecond(
        time=WinTime(2020, 1, 2, 3, 4, second),
        channels=list(channels),
    )


def test_win_roundtrip_multichannel():
    original = WinFile(
        format="win",
        seconds=[
            _second(
                s,
                [
                    ChannelSecond(0x0A01, 100, _samples(100, s)),
                    ChannelSecond(0x0A02, 100, _samples(100, 10 + s)),
                    ChannelSecond(0x0A03, 100, _samples(100, -s)),
                ],
            )
            for s in range(3)
        ],
    )
    raw = encode_win_bytes(original)
    decoded = decode_win_bytes(raw)
    assert decoded.format == "win"
    assert len(decoded.seconds) == 3
    assert decoded.seconds[0].time.year == 2020
    assert decoded.seconds[0].time.timezone == "Asia/Tokyo"
    for s in range(3):
        for i, cid in enumerate((0x0A01, 0x0A02, 0x0A03)):
            ch = decoded.seconds[s].channels[i]
            assert ch.channel_id == cid
            assert ch.sample_rate == 100
            np.testing.assert_array_equal(ch.samples, original.seconds[s].channels[i].samples)


def test_win32_roundtrip():
    samples = np.arange(100, dtype=np.int32) * 3 - 40
    original = WinFile(
        format="win32",
        seconds=[
            WinSecond(
                time=WinTime(2024, 12, 31, 23, 59, 58),
                channels=[ChannelSecond(0xABCD, 100, samples)],
            )
        ],
    )
    raw = encode_win_bytes(original)
    assert raw[:4] == b"\x00\x00\x00\x00"
    decoded = decode_win_bytes(raw)
    assert decoded.format == "win32"
    np.testing.assert_array_equal(decoded.seconds[0].channels[0].samples, samples)
    assert decoded.seconds[0].time.hour == 23


def test_large_diffs_use_wider_samples():
    samples = np.arange(50, dtype=np.int32) * 1000
    original = WinFile(
        format="win",
        seconds=[
            WinSecond(
                time=WinTime(1999, 5, 6, 7, 8, 9),
                channels=[ChannelSecond(1, 50, samples)],
            )
        ],
    )
    decoded = decode_win_bytes(encode_win_bytes(original))
    np.testing.assert_array_equal(decoded.seconds[0].channels[0].samples, samples)
    assert decoded.seconds[0].time.year == 1999


def test_full_range_i32_deltas_roundtrip():
    samples = np.array(
        [np.iinfo(np.int32).min, np.iinfo(np.int32).max, np.iinfo(np.int32).min, 0],
        dtype=np.int32,
    )
    original = WinFile(
        format="win",
        seconds=[
            WinSecond(
                time=WinTime(2020, 1, 2, 3, 4, 5),
                channels=[ChannelSecond(0x00FF, 4, samples)],
            )
        ],
    )
    decoded = decode_win_bytes(encode_win_bytes(original))
    np.testing.assert_array_equal(decoded.seconds[0].channels[0].samples, samples)


def test_decode_win_path(tmp_path):
    from win2zarr import decode_win

    original = WinFile(
        format="win",
        seconds=[
            WinSecond(
                time=WinTime(2020, 1, 1, 0, 0, 0),
                channels=[ChannelSecond(7, 20, np.arange(20, dtype=np.int32))],
            )
        ],
    )
    path = tmp_path / "tiny.win"
    path.write_bytes(encode_win_bytes(original))
    decoded = decode_win(path)
    assert decoded.source.endswith("tiny.win")
    np.testing.assert_array_equal(decoded.seconds[0].channels[0].samples, np.arange(20))
