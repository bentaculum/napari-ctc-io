from pathlib import Path

from napari_ctc_io import napari_get_reader


def test_reader():
    path = Path(__file__).parent / "resources" / "TRA"
    reader = napari_get_reader(path)
    assert callable(reader)

    layer_data_list = reader(path)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 2


if __name__ == "__main__":
    test_reader()
