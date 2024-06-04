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
    """Writes a labels and a tracks layer to CTC format.

    https://celltrackingchallenge.net/datasets/

    Args
        path: Directory to save to.
        data : A list of layer tuples.
            Tuples contain three elements: (data, meta, layer_type)
            `data` is the layer data
            `meta` is a dictionary containing all other metadata attributes
            from the napari layer (excluding the `.data` layer attribute).
            `layer_type` is a string, eg: "image", "labels", "surface", etc.

    Returns:
        list: A list containing (potentially multiple) string paths to the saved file(s).
    """
    logger.info(f"Writing CTC format to {path}")  # noqa: G004
    if len(data) != 2:
        raise ValueError("Need_two_layers to save.")

    masks = data[0]
    tracks = data[1]

    if masks[2] != "labels":
        raise ValueError("First layer needs to be a labels layer.")
    if tracks[2] != "tracks":
        raise ValueError("Second layer needs to be a tracks layer.")

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

    return [path]


def tracks_to_ctc(tracks: np.ndarray, graph: dict) -> pd.DataFrame:
    """Convert from napari.layers.Tracks to CTC format.

    Args:
        tracks: ID,T,(Z),Y,X.
        graph: id: [parent]
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
