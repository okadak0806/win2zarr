"""Minute-shard Zarr layout for analysis-ready WIN waveforms.

Golden-path store (100 Hz)::

    <out_dir>/<YYYYMMDDTHHMM>.zarr/
        waveforms    uint16  (n_samples, n_ch)  chunks (6000, 3)
        valid_mask   bool    (n_samples, n_ch)  chunks (6000, 3)
        .zattrs      sample_rate, timezone=Asia/Tokyo, start_time, ...

``waveforms`` holds IEEE bfloat16 bit patterns (not a native zarr dtype).
WIN BCD timestamps are Japan Standard Time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from win2zarr.bf16 import bf16_bits_to_f32, f32_to_bf16_bits
from win2zarr.channel_table import ChannelTable

PathLike = Union[str, Path]
JST = timezone(timedelta(hours=9), name="Asia/Tokyo")

COMPONENT_ORDER = {"U": 0, "Z": 0, "UD": 0, "VH": 0, "N": 1, "NS": 1, "E": 2, "EW": 2}


@dataclass
class MinuteShard:
    start: datetime
    sample_rate: float
    channel_ids: List[int]
    stations: List[str]
    components: List[str]
    waveforms: np.ndarray  # float32 (n_samples, n_ch)
    valid_mask: np.ndarray  # bool
    timezone: str = "Asia/Tokyo"
    units: Optional[List[Optional[str]]] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.waveforms.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.waveforms.shape[1])

    @property
    def shard_name(self) -> str:
        local = self.start.astimezone(JST)
        return local.strftime("%Y%m%dT%H%M") + ".zarr"


def _channel_sort_key(channel_id: int, table: Optional[ChannelTable]) -> Tuple:
    rec = table.get(channel_id) if table is not None else None
    if rec is None:
        return ("", 99, channel_id)
    comp = rec.component.upper()
    return (rec.station, COMPONENT_ORDER.get(comp, 50), rec.component, channel_id)


def _ordered_channel_ids(ids: Iterable[int], table: Optional[ChannelTable]) -> List[int]:
    unique = []
    seen = set()
    for cid in ids:
        cid = int(cid)
        if cid not in seen:
            unique.append(cid)
            seen.add(cid)
    unique.sort(key=lambda c: _channel_sort_key(c, table))
    return unique


def _station_component(channel_id: int, table: Optional[ChannelTable]) -> Tuple[str, str]:
    rec = table.get(channel_id) if table is not None else None
    if rec is None:
        return (f"{channel_id:04X}", "")
    return rec.station, rec.component


def plan_shards(
    files,
    sample_rate: float = 100.0,
    channel_table: Optional[ChannelTable] = None,
    physical: bool = True,
) -> List[MinuteShard]:
    """Group decoded seconds into minute shards at ``sample_rate`` Hz."""
    n_samples = int(round(sample_rate * 60.0))
    buckets: Dict[datetime, Dict[int, Dict[int, np.ndarray]]] = {}
    all_ids = []
    for win_file in files:
        for sec in win_file.seconds:
            start = sec.time.to_datetime().replace(second=0, microsecond=0)
            second_index = sec.time.second
            bucket = buckets.setdefault(start, {})
            for ch in sec.channels:
                if int(ch.sample_rate) != int(sample_rate):
                    continue
                all_ids.append(int(ch.channel_id))
                by_sec = bucket.setdefault(int(ch.channel_id), {})
                by_sec[second_index] = np.asarray(ch.samples, dtype=np.int32)

    channel_ids = _ordered_channel_ids(all_ids, channel_table)
    shards: List[MinuteShard] = []
    for start in sorted(buckets):
        waveforms = np.zeros((n_samples, len(channel_ids)), dtype=np.float32)
        valid = np.zeros((n_samples, len(channel_ids)), dtype=bool)
        bucket = buckets[start]
        stations = []
        components = []
        units: List[Optional[str]] = []
        for col, cid in enumerate(channel_ids):
            sta, comp = _station_component(cid, channel_table)
            stations.append(sta)
            components.append(comp)
            rec = channel_table.get(cid) if channel_table is not None else None
            scale = 1.0
            unit = "count"
            if physical and rec is not None and rec.counts_to_unit is not None:
                scale = float(rec.counts_to_unit)
                unit = rec.unit or "unknown"
            units.append(unit)
            for second_index, samples in bucket.get(cid, {}).items():
                i0 = int(second_index) * int(sample_rate)
                i1 = i0 + int(sample_rate)
                n = min(len(samples), i1 - i0)
                waveforms[i0 : i0 + n, col] = samples[:n].astype(np.float32) * np.float32(scale)
                valid[i0 : i0 + n, col] = True
        shards.append(
            MinuteShard(
                start=start,
                sample_rate=float(sample_rate),
                channel_ids=list(channel_ids),
                stations=stations,
                components=components,
                waveforms=waveforms,
                valid_mask=valid,
                units=units,
            )
        )
    return shards


def _open_group(path: Path, mode: str):
    import zarr

    path = Path(path)
    kwargs = {"mode": mode}
    try:
        return zarr.open_group(str(path), zarr_format=2, **kwargs)
    except TypeError:
        return zarr.open_group(str(path), **kwargs)


def _compressor():
    try:
        from numcodecs import Blosc

        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except Exception:  # pragma: no cover
        return None


def _create_array(group, name: str, data: np.ndarray, chunks, dtype):
    compressor = _compressor()
    if hasattr(group, "create_dataset"):
        return group.create_dataset(
            name,
            data=data,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            overwrite=True,
        )
    if name in group:
        del group[name]
    data = np.ascontiguousarray(data)
    kwargs = {
        "name": name,
        "data": data,
        "chunks": chunks,
        "overwrite": True,
    }
    if compressor is not None:
        try:
            return group.create_array(compressors=[compressor], **kwargs)
        except (TypeError, ValueError):
            pass
    return group.create_array(**kwargs)


def write_minute_shards(
    shards: Sequence[MinuteShard],
    out_dir: PathLike,
    overwrite: bool = True,
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for shard in shards:
        path = out_dir / shard.shard_name
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        _write_one(shard, path)
        written.append(path)
    return written


def _write_one(shard: MinuteShard, path: Path) -> None:
    n_samples, n_ch = shard.waveforms.shape
    chunk_samples = min(6000, n_samples) if n_samples else 6000
    chunk_ch = min(3, n_ch) if n_ch else 1
    chunks = (chunk_samples, chunk_ch)
    bits = f32_to_bf16_bits(shard.waveforms)
    group = _open_group(path, mode="w")
    _create_array(group, "waveforms", bits, chunks, np.uint16)
    _create_array(group, "valid_mask", shard.valid_mask.astype(np.bool_), chunks, np.bool_)
    start = shard.start.astimezone(JST)
    attrs = {
        "sample_rate": float(shard.sample_rate),
        "timezone": "Asia/Tokyo",
        "start_time": start.isoformat(),
        "n_samples": int(n_samples),
        "n_channels": int(n_ch),
        "channel_ids": [f"{cid:04X}" for cid in shard.channel_ids],
        "stations": list(shard.stations),
        "components": list(shard.components),
        "units": list(shard.units or []),
        "waveforms_dtype": "bfloat16_bits",
        "chunks": [int(chunk_samples), int(chunk_ch)],
        "win_bcd_time": "JST",
    }
    for key, value in attrs.items():
        group.attrs[key] = value


def open_zarr_store(path: PathLike) -> MinuteShard:
    import zarr

    path = Path(path)
    try:
        group = zarr.open_group(str(path), mode="r", zarr_format=2)
    except TypeError:
        group = zarr.open_group(str(path), mode="r")
    bits = np.asarray(group["waveforms"][:], dtype=np.uint16)
    waveforms = bf16_bits_to_f32(bits)
    valid = np.asarray(group["valid_mask"][:], dtype=bool)
    attrs = dict(group.attrs)
    start_raw = attrs.get("start_time")
    if isinstance(start_raw, str):
        start = datetime.fromisoformat(start_raw)
        if start.tzinfo is None:
            start = start.replace(tzinfo=JST)
    else:
        start = datetime(1970, 1, 1, tzinfo=JST)
    channel_ids = [int(x, 16) if isinstance(x, str) else int(x) for x in attrs.get("channel_ids", [])]
    return MinuteShard(
        start=start,
        sample_rate=float(attrs.get("sample_rate", 100.0)),
        channel_ids=channel_ids,
        stations=list(attrs.get("stations", [])),
        components=list(attrs.get("components", [])),
        waveforms=waveforms,
        valid_mask=valid,
        timezone=str(attrs.get("timezone", "Asia/Tokyo")),
        units=list(attrs.get("units", [])),
        attrs=attrs,
    )
