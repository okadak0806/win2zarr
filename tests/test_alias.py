import win2zarr
import win_loader


def test_win_loader_alias():
    assert win_loader.decode_win is win2zarr.decode_win
    assert win_loader.encode_win_bytes is win2zarr.encode_win_bytes
    assert win_loader.__version__ == win2zarr.__version__
