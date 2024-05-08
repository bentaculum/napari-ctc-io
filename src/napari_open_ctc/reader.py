"""
TODOs
- add tests
- add progress bar in the viewer: https://github.com/napari/napari/blob/main/examples/progress_bar_minimal_.py
- sync colors of tracking and segmentation layers
- put in the color-by-child layer as well, invisible
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
from tifffile import imread
from tqdm import tqdm


def ctc_reader(path: Path) -> Callable:
    segmentation, man_track = _load_tra(path)
    tracks, tracks_graph = _ctc_to_napari_tracks(segmentation, man_track)
    print(tracks.shape)
    _check_ctc(
        tracks=pd.DataFrame(man_track).astype(int),
        detections=pd.DataFrame(tracks[:, :2]).astype(int),
        masks=segmentation,
    )
    return lambda path: [
        (segmentation, dict(name="labels"), "labels"),  # noqa
        (tracks, dict(graph=tracks_graph, name="tracks"), "tracks"),  # noqa
    ]


def _ctc_to_napari_tracks(
    segmentation: np.ndarray, man_track: np.ndarray
) -> tuple[np.ndarray, dict]:
    """convert a traccuracy graph to napari tracks"""

    tracks = []
    for t, frame in tqdm(
        enumerate(segmentation),
        total=len(segmentation),
        leave=True,
        desc="Computing centroids",
    ):
        props = pd.DataFrame(
            regionprops_table(frame, properties=("label", "centroid"))
        )
        props.insert(1, "t", t)
        tracks.append(props)

    tracks = pd.concat(tracks).to_numpy()

    tracks_graph = {}
    for idx, _, _, parent in tqdm(
        man_track,
        desc="Converting CTC to napari tracks",
        leave=False,
    ):
        if parent != 0:
            tracks_graph[idx] = [parent]

    return tracks, tracks_graph


class FoundTracks(Exception):
    pass


def _load_tra(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load segmentation and tracks from a folder.

    Checks for
    - alphanumerically ordered *.tif files
    - tracks in txt format, preferably called `man_track.txt` or `res_track.txt`.

    Args:
        path (Path): Folder with segmentations and tracks.

    Raises:
        ValueError:

    Returns:
        np.ndarray: stacked segmentations as T x 2D/3D array
        np.ndarray: tracks as N x 4 array
    """
    tracks_globs = ["man_track.txt", "res_track.txt", "*.txt"]

    try:
        for _glob in tracks_globs:
            print(f"Trying to load tracks with `glob {path / _glob}`")
            for fpath in path.glob(_glob):
                if fpath.exists():
                    tracks = np.loadtxt(fpath, delimiter=" ").astype(int)
                    print(fpath, tracks.shape)
                    raise FoundTracks
    except FoundTracks:
        print(f"Loaded tracks from {fpath}")
    else:
        raise ValueError(f"Did not find a .txt file with tracks in {path}.")

    files = sorted(path.glob("*.tif"))
    segmentation = np.stack(
        [
            imread(x)
            for x in tqdm(
                files,
                desc="Loading segmentations",
                leave=True,
            )
        ]
    )

    return segmentation, tracks


def _check_ctc(
    tracks: pd.DataFrame,
    detections: pd.DataFrame,
    masks: np.ndarray,
):
    """Sanity checks for valid CTC format.

    Adapted from https://github.com/Janelia-Trackathon-2023/traccuracy/blob/main/src/traccuracy/loaders/_ctc.py

    Hard checks (throws exception):
    - Tracklet IDs in tracks file must be unique and positive
    - Parent tracklet IDs must exist in the tracks file
    - Intertracklet edges must be directed forward in time.
    - In each time point, the set of segmentation IDs present in the detections must equal the set
    of tracklet IDs in the tracks file that overlap this time point.

    Soft checks (prints warning):
    - No duplicate tracklet IDs (non-connected pixels with same ID) in a single timepoint.

    Args:
        tracks (pd.DataFrame): Tracks in CTC format with columns Cell_ID, Start, End, Parent_ID.
        detections (pd.DataFrame): Detections extracted from masks, containing columns
            segmentation_id, t.
        masks (np.ndarray): Set of masks with time in the first axis.
    Raises:
        ValueError: If any of the hard checks fail.
    """

    print("Running CTC format checks")
    tracks.columns = ["Cell_ID", "Start", "End", "Parent_ID"]
    detections.columns = ["segmentation_id", "t"]

    if tracks["Cell_ID"].min() < 1:
        raise ValueError("Cell_IDs in tracks file must be positive integers.")
    if len(tracks["Cell_ID"]) < len(tracks["Cell_ID"].unique()):
        raise ValueError("Cell_IDs in tracks file must be unique integers.")

    for _, row in tqdm(
        tracks.iterrows(),
        desc="Checking parent links",
    ):
        if row["Parent_ID"] != 0:
            if row["Parent_ID"] not in tracks["Cell_ID"].values:
                raise ValueError(
                    f"Parent_ID {row['Parent_ID']} is not present in tracks."
                )
            parent_end = tracks[tracks["Cell_ID"] == row["Parent_ID"]][
                "End"
            ].iloc[0]
            if parent_end >= row["Start"]:
                raise ValueError(
                    f"Invalid tracklet connection: Daughter tracklet with ID {row['Cell_ID']} "
                    f"starts at t={row['Start']}, "
                    f"but parent tracklet with ID {row['Parent_ID']} only ends at t={parent_end}."
                )

    for t in tqdm(
        range(tracks["Start"].min(), tracks["End"].max()),
        desc="Checking missing IDs",
    ):
        track_ids = set(
            tracks[(tracks["Start"] <= t) & (tracks["End"] >= t)]["Cell_ID"]
        )
        det_ids = set(detections[(detections["t"] == t)]["segmentation_id"])
        if not track_ids.issubset(det_ids):
            raise ValueError(
                f"Missing IDs in masks at t={t}: {track_ids - det_ids}"
            )
        if not det_ids.issubset(track_ids):
            raise ValueError(
                f"IDs {det_ids - track_ids} at t={t} not represented in tracks file."
            )

    for t, frame in tqdm(
        enumerate(masks),
        desc="Checking for non-connected masks",
    ):
        _, n_components = label(frame, return_num=True)
        n_labels = len(detections[detections["t"] == t])
        if n_labels < n_components:
            print(f"{n_components - n_labels} non-connected masks at t={t}.")

    print("Checks completed")
