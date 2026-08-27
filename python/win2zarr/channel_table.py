"""WIN / Hi-net channel table (``.ch``) parser.

Supports the two common on-disk dialects:

* Hi-net / Nagoya layout: station at column 4, component at column 5.
* Classic ERI WIN layout: an extra numeric field before the station name
  (station at column 5, component at column 6).

``*`` placeholders are treated as missing. Lines starting with ``#`` are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

PathLike = Union[str, Path]


def _is_number(token: str) -> bool:
    if token in {"*", "-"}:
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def _opt_float(token: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if token is None or token in {"*", ""}:
        return default
    try:
        return float(token)
    except ValueError:
        return default


def _opt_int(token: Optional[str], default: Optional[int] = None) -> Optional[int]:
    value = _opt_float(token, None)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True)
class ChannelRecord:
    channel_id: int
    station: str
    component: str
    flag: Optional[int] = None
    delay_ms: Optional[float] = None
    sensitivity: Optional[float] = None
    unit: Optional[str] = None
    period: Optional[float] = None
    damping: Optional[float] = None
    amplification_db: Optional[float] = None
    step_v: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None

    @property
    def hex_id(self) -> str:
        return f"{self.channel_id:04X}"

    @property
    def counts_to_unit(self) -> Optional[float]:
        """Integer counts → physical unit, following the published WIN table formula.

        ``step / (sensitivity * 10**(ampl_db / 20))``
        """
        if self.step_v is None or self.sensitivity in (None, 0.0):
            return None
        ampl = 0.0 if self.amplification_db is None else self.amplification_db
        return self.step_v / (self.sensitivity * (10.0 ** (ampl / 20.0)))


class ChannelTable:
    def __init__(self, records: Iterable[ChannelRecord]):
        self.records: List[ChannelRecord] = list(records)
        self._by_id: Dict[int, ChannelRecord] = {r.channel_id: r for r in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def get(self, channel_id: int) -> Optional[ChannelRecord]:
        return self._by_id.get(int(channel_id))

    def stations(self) -> List[str]:
        seen = []
        for rec in self.records:
            if rec.station not in seen:
                seen.append(rec.station)
        return seen


def _parse_line(line: str) -> Optional[ChannelRecord]:
    raw = line.split("#", 1)[0].strip()
    if not raw:
        return None
    fields = raw.split()
    if len(fields) < 5:
        return None
    try:
        channel_id = int(fields[0], 16)
    except ValueError:
        return None

    # Autodetect dialect: extra numeric column before station (ERI) vs Hi-net.
    if len(fields) >= 6 and _is_number(fields[3]) and not _is_number(fields[4]):
        station = fields[4]
        component = fields[5]
        rest = fields[6:]
        # ERI layout then skips two more fields (monitor / ad-bits) before sensitivity.
        if len(rest) >= 4 and _is_number(rest[0]) and _is_number(rest[1]):
            rest = rest[2:]
        flag = _opt_int(fields[1])
        delay_ms = _opt_float(fields[2])
    else:
        station = fields[3]
        component = fields[4]
        rest = fields[5:]
        flag = _opt_int(fields[1]) if len(fields) > 1 else None
        delay_ms = _opt_float(fields[2]) if len(fields) > 2 else None

    # rest: [monitor?, ad_bits?,] sensitivity, unit, period, damp, ampl, step, lat, lon, elev
    # After dialect split, Hi-net rest starts at monitor index; skip two numeric prefixes
    # when they look like monitor-exp + AD bits.
    if len(rest) >= 4 and _is_number(rest[0]) and _is_number(rest[1]):
        rest = rest[2:]

    sensitivity = _opt_float(rest[0]) if len(rest) > 0 else None
    unit = None if len(rest) < 2 or rest[1] in {"*", ""} else rest[1]
    period = _opt_float(rest[2]) if len(rest) > 2 else None
    damping = _opt_float(rest[3]) if len(rest) > 3 else None
    amplification_db = _opt_float(rest[4]) if len(rest) > 4 else None
    step_v = _opt_float(rest[5]) if len(rest) > 5 else None
    latitude = _opt_float(rest[6]) if len(rest) > 6 else None
    longitude = _opt_float(rest[7]) if len(rest) > 7 else None
    elevation = _opt_float(rest[8]) if len(rest) > 8 else None

    return ChannelRecord(
        channel_id=channel_id,
        station=station,
        component=component,
        flag=flag,
        delay_ms=delay_ms,
        sensitivity=sensitivity,
        unit=unit,
        period=period,
        damping=damping,
        amplification_db=amplification_db,
        step_v=step_v,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
    )


def parse_channel_table(path: PathLike) -> ChannelTable:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    records = []
    for line in text.splitlines():
        rec = _parse_line(line)
        if rec is not None:
            records.append(rec)
    return ChannelTable(records)


def parse_channel_table_text(text: str) -> ChannelTable:
    records = []
    for line in text.splitlines():
        rec = _parse_line(line)
        if rec is not None:
            records.append(rec)
    return ChannelTable(records)
