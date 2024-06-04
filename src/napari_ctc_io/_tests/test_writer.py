from pathlib import Path

import numpy as np
from tifffile import imread

from napari_ctc_io import napari_get_reader, write_multiple


def test_writer():
    path = Path(__file__).parent / "resources" / "TRA"
    reader = napari_get_reader(path)
    assert callable(reader)

    layer_data_list = reader(path)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 2

    out_path = path.parent / "TRA_out"
    write_multiple(str(out_path), data=layer_data_list)

    # Compare all files
    for p_in, p_out in zip(
        sorted(path.glob("*.tif")), sorted(out_path.glob("*.tif"))
    ):
        _in = imread(p_in)
        _out = imread(p_out)

        assert np.all(_in == _out)

    in_tracks = np.loadtxt(list(path.glob("*.txt"))[0], delimiter=" ").astype(
        int
    )
    out_tracks = np.loadtxt(
        list(out_path.glob("*.txt"))[0], delimiter=" "
    ).astype(int)

    assert np.all(in_tracks == out_tracks)


if __name__ == "__main__":
    test_writer()
