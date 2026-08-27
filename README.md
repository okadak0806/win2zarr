# win2zarr

Convert [WIN](https://wwweic.eri.u-tokyo.ac.jp/WIN/man.en/winformat.html) /
WIN32 seismic waveform files to analysis-ready [Zarr](https://zarr.dev/) stores.

The decoder is an independent Rust implementation of the **published** WIN/WIN32
waveform format (ERI `winformat(1W)` on-disk layout). It is **not** a copy of
the Earthquake Research Institute WIN package sources (`winlib.c` and related
GPL-2.0 code).

WIN BCD timestamps are Japan Standard Time (`Asia/Tokyo`).

## Install

Python ≥ 3.9, a Rust toolchain, and a C compiler (for the PyO3 extension):

```bash
pip install -e ".[zarr]"
# or, from a clone without an isolated build env:
maturin develop --extras zarr
```

A third party can also install directly from GitHub:

```bash
pip install "win2zarr[zarr] @ git+https://github.com/okadak0806/win2zarr.git"
```

Optional extra `zarr` pulls in `zarr` and `numcodecs`. Decode-only use does not
need it.

## CLI

```bash
win-to-zarr --help
win-to-zarr plan   path/to/data.win --ch stations.ch --out ./zarr
win-to-zarr convert path/to/data.win --ch stations.ch --out ./zarr --yes
win-to-zarr convert path/to/data.win --out ./zarr --dry-run
```

`--yes` skips the write confirmation. `--dry-run` prints the plan and exits.
`--counts` stores integer counts as float instead of applying the channel-table
counts→physical conversion.

## Python API

```python
import numpy as np
import win2zarr

win = win2zarr.decode_win("example.win")
paths = win2zarr.write_zarr(win, "out", channel_table="stations.ch")

shard = win2zarr.open_zarr(paths[0])
wave = shard.waveforms          # float32, expanded from bfloat16 bits
mask = shard.valid_mask         # bool, False where that second was missing
print(shard.attrs["sample_rate"], shard.attrs["timezone"])
```

`import win_loader` is kept as a thin alias of `win2zarr`.

Synthetic WIN bytes (no field recordings) for tests or demos:

```python
from win2zarr import ChannelSecond, WinFile, WinSecond, WinTime, encode_win_bytes, decode_win_bytes

samples = (np.arange(100, dtype=np.int32) % 7) - 3
raw = encode_win_bytes(
    WinFile(
        format="win",
        seconds=[
            WinSecond(
                time=WinTime(2020, 1, 1, 0, 0, 0),
                channels=[ChannelSecond(0x0A01, 100, samples)],
            )
        ],
    )
)
assert decode_win_bytes(raw).seconds[0].channels[0].samples.tolist() == samples.tolist()
```

## Zarr layout

Golden-path store: **one Zarr group per JST minute**.

```
<out>/<YYYYMMDDTHHMM>.zarr/
  waveforms    uint16  (n_samples, n_channels)  chunks (6000, 3)
  valid_mask   bool    (n_samples, n_channels)  chunks (6000, 3)
  .zattrs
    sample_rate   100.0
    timezone      "Asia/Tokyo"
    start_time    ISO-8601 JST
    channel_ids   ["0A01", ...]
    stations      [...]
    components    [...]
    waveforms_dtype  "bfloat16_bits"
```

At 100 Hz a minute is 6000 samples. `waveforms` stores **bfloat16 bit patterns**
as `uint16` (widely portable; zarr/numcodecs have no native `bfloat16`).
`valid_mask` is false for missing seconds/channels. Channels are ordered by
station, then component (`U`/`Z`, `N`, `E`, others) so the `(6000, 3)` chunk
fits three-component stations.

## Channel table

A WIN `.ch` file maps 16-bit channel IDs to station/component and (optionally)
the counts→physical conversion. Hi-net-style tables (station in column 4) and
classic ERI tables (extra numeric field before the station name) are both
accepted. `#` comments and `*` placeholders are skipped.

## What WIN is

WIN is a multi-channel earthquake waveform format used across Japanese academic
and JMA/Hi-net data paths. On disk, a file is a concatenation of **second
blocks**: a 4-byte big-endian size, a 6-byte BCD timestamp, then per-channel
blocks (channel ID, 4-bit sample width + 12-bit sample rate, a 4-byte first
sample, then compressed first-differences). WIN32 (Hi-net `.cnt`) uses a
4-byte zero file prefix, a 16-byte second header with a 4-digit BCD year, and
two extra organisation/network bytes in front of each channel.

Format reference: [ERI winformat manpage](https://wwweic.eri.u-tokyo.ac.jp/WIN/man.en/winformat.html).

## License

Apache License 2.0. Copyright 2026 Kazumi Okada. See [`LICENSE`](LICENSE).

This repository does **not** include:

- ERI WIN package C sources (`src/winlib.c`, `COPYING`, GPL-2.0)
- real Hi-net / JMA / field recordings (those are not redistributable)
- ingest pipelines (ClickHouse, MQTT, wdisk, PhaseNet, …)

Tests use **synthetic WIN bytes only**.
