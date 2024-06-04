from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from napari.utils import progress
from tifffile import imwrite

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def write_multiple(path: str, data: List[FullLayerData]) -> List[str]:
    """Writes multiple layers of different types.

    Parameters
    ----------
    path : str
        A string path indicating where to save the data file(s).
    data : A list of layer tuples.
        Tuples contain three elements: (data, meta, layer_type)
        `data` is the layer data
        `meta` is a dictionary containing all other metadata attributes
        from the napari layer (excluding the `.data` layer attribute).
        `layer_type` is a string, eg: "image", "labels", "surface", etc.

    Returns
    -------
    [path] : A list containing (potentially multiple) string paths to the saved file(s).
    """
    logger.debug("Writing to CTC format")
    if len(data) != 2:
        raise ValueError("Need_two_layers to save.")

    # TODO input checks

    masks = data[0]
    tracks = data[1]
    # import ipdb

    # ipdb.set_trace()

    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)

    n_digits = int(math.log10(len(masks[0]))) + 1
    for i, m in progress(enumerate(masks[0]), desc=f"Writing {masks[2]}"):
        imwrite(
            p / f"mask{i:0{n_digits}d}.tif",
            m,
            compression="zstd",
        )

    res_track = tracks_to_ctc(tracks[0], tracks[1]["graph"])
    res_track.to_csv(
        p / "res_track.txt",
        index=False,
        header=None,
        sep=" ",
    )

    # return path to any file(s) that were successfully written
    return [path]


def napari_write_labels(path: str, data: List[FullLayerData]) -> List[str]:
    """Writes multiple layers of different types.

    Parameters
    ----------
    path : str
        A string path indicating where to save the data file(s).
    data : A list of layer tuples.
        Tuples contain three elements: (data, meta, layer_type)
        `data` is the layer data
        `meta` is a dictionary containing all other metadata attributes
        from the napari layer (excluding the `.data` layer attribute).
        `layer_type` is a string, eg: "image", "labels", "surface", etc.

    Returns
    -------
    [path] : A list containing (potentially multiple) string paths to the saved file(s).
    """
    if len(data) != 1:
        print("Need two layers to save.")

    # if data[0][2]
    labels = data[0]
    print("Using plugin writer")
    # tracks = data[1]

    imwrite(path, labels[0])

    # res_track = tracks_to_ctc(tracks[0], tracks[1]["graph"])

    # return path to any file(s) that were successfully written
    return [path]


def tracks_to_ctc(tracks: np.ndarray, graph: dict) -> pd.DataFrame:
    """

    Args:
        tracks: ID,T,(Z),Y,X.
        graph (_type_): _description_

    Returns:
        _type_: _description_
    """
    rows = []
    # TODO Allow for gap closing

    for _id in progress(np.unique(tracks[:, 0])):
        timepoints = tracks[tracks[:, 0] == _id][:, 1]
        start = timepoints.min()
        end = timepoints.max()
        if end - start > len(timepoints) - 1:
            raise RuntimeError(f"Track {_id} has one or multiple gaps.")
        if end - start < len(timepoints) - 1:
            raise RuntimeError(
                f"Track {_id} has too many entries in napari.layers.Tracks."
            )

        row = {
            "id": _id,
            "start": start,
            "end": end,
            "parent": graph.get(_id, [0])[0],
        }
        rows.append(row)
    df = pd.DataFrame(rows).astype(int)

    return df
