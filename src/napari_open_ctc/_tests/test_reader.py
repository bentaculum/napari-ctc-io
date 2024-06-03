from pathlib import Path

from napari_open_ctc import napari_get_reader


def test_reader():
    path = Path(__file__).parent / "resources" / "TRA"
    reader = napari_get_reader(path)
    assert callable(reader)

    layer_data_list = reader(path)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 2


# def test_get_reader_pass():
#     reader = napari_get_reader("fake.file")
#     assert reader is None

if __name__ == "__main__":
    test_reader()
