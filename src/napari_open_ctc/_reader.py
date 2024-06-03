from pathlib import Path

from .reader import ctc_reader


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # open multiple folders
        raise NotImplementedError("Open multiple folders at once")

    # if we know we cannot read the file, we immediately return None.
    path = Path(path)
    if not path.is_dir():
        return None

    # otherwise we return the *function* that can read ``path``.
    return ctc_reader(path)
