from win2zarr.channel_table import parse_channel_table_text


HINET = """\
# Hi-net / Nagoya layout: station is column 4
0A01 1 0 N.ABC U 7 24 800 m/s 1.0 0.7 20 2.44e-6 35.1 136.9 56 0 0
0A02 1 0 N.ABC N 7 24 800 m/s 1.0 0.7 20 2.44e-6 35.1 136.9 56 0 0
0A03 1 0 N.ABC E 7 24 800 m/s 1.0 0.7 20 2.44e-6 35.1 136.9 56 0 0
0B01 1 0 N.XYZ U 7 24 * m/s * * * * 34.0 135.0 10
"""

ERI = """\
# Classic ERI: extra numeric field before station
1001 1 0 0 STA1 UD 0 0 200 m/s 1.0 0.7 0 1e-6 35.0 139.0 12
1002 1 0 0 STA1 NS 0 0 200 m/s 1.0 0.7 0 1e-6 35.0 139.0 12
"""


def test_hinet_layout():
    table = parse_channel_table_text(HINET)
    rec = table.get(0x0A01)
    assert rec is not None
    assert rec.station == "N.ABC"
    assert rec.component == "U"
    assert rec.sensitivity == 800
    assert rec.unit == "m/s"
    assert rec.counts_to_unit is not None
    assert rec.counts_to_unit > 0
    assert table.get(0x0B01).station == "N.XYZ"


def test_eri_layout():
    table = parse_channel_table_text(ERI)
    rec = table.get(0x1001)
    assert rec.station == "STA1"
    assert rec.component == "UD"
    assert table.stations() == ["STA1"]


def test_file_roundtrip(tmp_path):
    from win2zarr import parse_channel_table

    path = tmp_path / "stations.ch"
    path.write_text(HINET, encoding="utf-8")
    table = parse_channel_table(path)
    assert len(table) == 4
    assert table.get(0x0A03).component == "E"
