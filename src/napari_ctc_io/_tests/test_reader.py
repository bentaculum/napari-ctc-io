from pathlib import Path

import pytest

from napari_ctc_io import napari_get_reader


@pytest.mark.parametrize(
    "path",
    [
        Path(__file__).parent / "resources" / "fluo_simulated" / "TRA",
        Path(__file__).parent / "resources" / "edge_cases" / "TRA",
    ],
)
def test_reader(path):
    reader = napari_get_reader(path)
    assert callable(reader)

    layer_data_list = reader(path)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 2


if __name__ == "__main__":
    test_reader()
