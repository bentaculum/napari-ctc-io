from pathlib import Path
from typing import Union

import motile
import networkx as nx
import numpy as np
import pandas as pd
import traccuracy
from skimage.measure import regionprops
from tqdm import tqdm
from traccuracy.loaders import load_ctc_data as _load_ctc_data


def ctc_reader(path):
    """
    TODOs
    - Get rid of the motile dependency
    - add progress bar in the viewer: https://github.com/napari/napari/blob/main/examples/progress_bar_minimal_.py
    - sync colors of tracking and segmentation layers
    - put in the color-by-child layer as well, invisible
    """

    graph = load_ctc_graph(path)
    tracks, _ = graph_to_napari_tracks(graph.tracking_graph)
    tracks = tracks.astype(int)
    return lambda path: [
        (graph.segmentation, dict(name="labels"), "labels"),  # noqa
        (tracks, dict(name="tracks"), "tracks"),  # noqa
    ]


def load_ctc_graph(
    path: Union[str, Path]
) -> traccuracy._tracking_data.TrackingData:
    """load a ctc dataset and convert it to traccuracy TrackingData
    (that contains the tracking graph and the segmentation data)

    Returns:
        traccuracy._tracking_data.TrackingData: tracking data with graph and segmentation
    """
    path = Path(path)
    man_track = path / "man_track.txt"
    if not man_track.exists():
        raise ValueError(f"{path} doesnt contain the file man_track.txt !")
    graph = _load_ctc_data(path, man_track)

    return graph


def traccuracy_to_motile(
    graph: traccuracy._tracking_data.TrackingGraph,
) -> motile.TrackGraph:
    """convert traccuracy graph to motile graph by adjusting some attributes"""
    nodes = graph.nodes()
    edges = graph.edges()
    nx_graph = nx.DiGraph()
    for n in nodes:
        attrs = nodes[n].copy()
        attrs["time"] = attrs["t"]
        coords = tuple(attrs.get(t, None) for t in "xyz")
        coords = np.array(
            tuple(c for c in coords if c is not None), dtype=np.float32
        )
        attrs["coords"] = coords
        attrs["label"] = attrs["segmentation_id"]
        attrs["id"] = n
        attrs["index"] = -1  # dummy index
        for t in "xyz":
            if t in attrs:
                del attrs[t]
        del attrs["t"]
        del attrs["segmentation_id"]
        nx_graph.add_node(n, **attrs)
    for e in edges:
        nx_graph.add_edge(*e, **edges[e])

    motile_graph = motile.TrackGraph(frame_attribute="time")
    motile_graph.add_from_nx_graph(nx_graph)
    return motile_graph


def linear_chains(G: nx.DiGraph):
    """find all linear chains in a tree/graph, i.e. paths that

    i) either start/end at a node with out_degree>in_degree or and have no internal branches, or
    ii) consists of a single node

    note that each chain includes its start/end node, i.e. they can be appear in multiple chains
    """
    # get all nodes with out_degree>in_degree (i.e. start of chain)
    nodes = tuple(n for n in G.nodes if G.out_degree[n] > G.in_degree[n])
    single_nodes = tuple(
        n for n in G.nodes if G.out_degree[n] == G.in_degree[n] == 0
    )

    for ni in single_nodes:
        yield [ni]

    for ni in nodes:
        neighs = tuple(G.neighbors(ni))
        for child in neighs:
            path = [ni, child]
            while len(childs := tuple(G.neighbors(path[-1]))) == 1:
                path.append(childs[0])
            yield path


def _motile_to_nx(graph: motile.TrackGraph):
    """convert motile graph to networkx graph"""
    G = nx.DiGraph()
    G.add_nodes_from(graph.nodes.keys())
    G.add_edges_from(graph.edges.keys())
    return G


def graph_to_napari_tracks(
    graph: Union[motile.TrackGraph, traccuracy._tracking_graph.TrackingGraph]
):
    """convert a traccuracy graph to napari tracks"""
    if isinstance(graph, traccuracy._tracking_graph.TrackingGraph):
        print("converting traccuracy graph to motile graph")
        graph = traccuracy_to_motile(graph)
    elif isinstance(graph, motile.TrackGraph):
        pass
    else:
        raise ValueError(
            f"graph must be of type traccuracy._tracking_graph.TrackingGraph or motile.TrackGraph, but is {type(graph)}"
        )

    # convert to networkx graph
    G = _motile_to_nx(graph)

    # each tracklet is a linear chain in the graph
    chains = tuple(linear_chains(G))

    track_end_to_track_id = {}
    labels = []
    for i, cs in enumerate(chains):
        label = i + 1
        labels.append(label)
        # start nodes of dividing tracks include the parent
        start, end = cs[0], cs[-1]
        track_end_to_track_id[end] = label

    tracks = []
    tracks_graph = {}

    for label, cs in tqdm(zip(labels, chains), total=len(chains)):
        start = cs[0]
        if start in track_end_to_track_id:
            tracks_graph[label] = track_end_to_track_id[start]

        for c in cs:
            node = graph.nodes[c]
            t = node["time"]
            coord = node["coords"]
            tracks.append([label, t] + list(coord))

    tracks = np.array(tracks)
    return tracks, tracks_graph


def _check_ctc_df(df: pd.DataFrame, masks: np.ndarray):
    """sanity check of all labels in a CTC dataframe are present in the masks"""
    for t in tqdm(
        range(df.t1.min(), df.t1.max()), leave=False, desc="Checking CTC"
    ):
        sub = df[(df.t1 <= t) & (df.t2 >= t)]
        sub_lab = set(sub.label)
        masks_lab = set(np.unique(masks[t])) - {0}
        if not sub_lab.issubset(masks_lab):
            print(f"Missing labels in masks at t={t}: {sub_lab-masks_lab}")
            return False
    return True


def graph_to_ctc(
    graph: motile.TrackGraph,
    masks_original: list[np.ndarray],
    check: bool = True,
):
    """convert graph to ctc track Dataframe and relabeled masks

    Parameters:
    -----------
    graph: motile.TrackGraph
        motile track graph with node attributes "time" and "label"
    masks_original: list[np.ndarray]


    Returns:
    --------
    df: pd.DataFrame
        track dataframe with columns ['track_id', 't_start', 't_end', 'parent_id']
    masks: list[np.ndarray]
        list of masks with unique colors for each track
    """

    # convert to networkx graph
    G = _motile_to_nx(graph)

    # each tracklet is a linear chain in the graph
    chains = tuple(linear_chains(G))

    # special case for nodes that appear and directly divide
    div_parents = tuple(
        [n]
        for n in G.nodes
        if n in G.nodes and G.in_degree(n) == 0 and G.out_degree(n) > 1
    )
    chains = div_parents + chains

    rows = []

    parent_ids = {-1: 0}

    regions = tuple(
        {reg.label: reg.slice for reg in regionprops(m)}
        for t, m in enumerate(masks_original)
    )

    masks = [np.zeros_like(m) for m in masks_original]

    for i, chain in tqdm(
        enumerate(chains),
        total=len(chains),
        desc="Converting graph to CTC results",
    ):
        label = i + 1

        # remove parent if it is dividing
        if len(chain) > 1 and G.out_degree[chain[0]] > 1:
            chain = chain[1:]

        start, end = chain[0], chain[-1]

        t1 = graph.nodes[start]["time"]
        t2 = graph.nodes[end]["time"]

        parent_ids[end] = label

        parents = tuple(G.predecessors(start))
        if len(parents) == 0:
            parent = -1  # special case
        elif len(parents) == 1:
            parent = parents[0]
        else:
            raise ValueError("More than one parent!")

        # relabel masks
        for c in chain:
            node = graph.nodes[c]
            t = node["time"]
            lab = node["label"]
            ss = regions[t][lab]
            m = masks_original[t][ss] == lab
            if masks[t][ss][m].max() > 0:
                raise RuntimeError(f"Overlapping masks at t={t}, label={lab}")
            if np.count_nonzero(m) == 0:
                raise RuntimeError(f"Empty mask at t={t}, label={lab}")
            masks[t][ss][m] = label

        rows.append([label, t1, t2, parent])

    for r in rows:
        r[-1] = parent_ids[r[-1]]

    df = pd.DataFrame(rows, columns=["label", "t1", "t2", "parent"])

    masks = np.stack(masks)

    if check:
        _check_ctc_df(df, masks)

    return df, masks
