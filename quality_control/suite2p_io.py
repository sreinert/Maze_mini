"""
Shared helpers for the suite2p QC scripts: finding plane folders, naming them,
read-only loading and the figure colours.

Suite2p output sits in a preprocessed tree, e.g.

    /Volumes/behrens/Sandra/mini2p/fscan/test2/preprocessed/suite2p/plane0

A "session" here is the folder holding suite2p/ (test2/preprocessed above); its
plane0, plane1, ... folders are the planes. Labels drop the generic layout
folders, so the plane above is 'test2' and a second plane 'test2/plane1'. If
folder names follow sub-*/ses-*/run-*, those parts are used instead
(e.g. 'sub-01/ses-03/run-02').
"""
import os
import re

import numpy as np

PLANE_DIR_RE = re.compile(r"^plane\d+$")
LAYOUT_DIRS = {".", "preprocessed", "derivatives", "funcimg", "suite2p"}
BIDS_PART_RE = re.compile(r"^(sub|ses|run)-")

DEFAULT_OUT_ROOT = "~/Desktop/maze_mini_qc"

# dataviz reference palette (light mode), as in the Bonsai_ABCD QC scripts
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"


def find_plane_dirs(root, required="F.npy"):
    """All suite2p plane folders (plane0, plane1, ...) below root that contain
    `required` (a file name, or a tuple of names of which any one is enough).

    root may itself be a plane folder. Hidden folders and backup folders
    (names starting with 'backup') are skipped.
    """
    required = (required,) if isinstance(required, str) else tuple(required)
    root = os.path.abspath(os.path.expanduser(root))
    if PLANE_DIR_RE.match(os.path.basename(root)) and any(
            os.path.exists(os.path.join(root, r)) for r in required):
        return [root]
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and not d.lower().startswith("backup"))
        if PLANE_DIR_RE.match(os.path.basename(dirpath)) and any(r in filenames for r in required):
            found.append(dirpath)
            dirnames[:] = []
    return found


def session_label(plane_dir, root):
    """Label of the session a plane belongs to (see module docstring)."""
    parts = os.path.normpath(plane_dir).split(os.sep)
    tags = [p.split("_")[0] for p in parts if BIDS_PART_RE.match(p)]
    if tags:
        return "/".join(tags)
    rel = os.path.relpath(os.path.dirname(plane_dir), root)
    tags = [p for p in rel.split(os.sep) if p not in LAYOUT_DIRS]
    if not tags or tags[0] == "..":
        # root is the plane folder (or inside the layout folders): name it
        # after the first non-layout folder above the plane
        above = [p for p in parts[:-1] if p not in LAYOUT_DIRS]
        tags = above[-1:] or [parts[-1]]
    return "/".join(tags)


def plane_label(plane_dir, root):
    """session_label, plus the plane name for planes other than plane0."""
    label = session_label(plane_dir, root)
    plane = os.path.basename(plane_dir)
    return label if plane == "plane0" else f"{label}/{plane}"


def default_out_dir(root, kind):
    """~/Desktop/maze_mini_qc/<name of root>/<kind>"""
    return os.path.join(DEFAULT_OUT_ROOT, os.path.basename(os.path.abspath(root)), kind)


def load_npy(path, allow_pickle=False):
    """np.load for suite2p inputs. Arrays are memory-mapped read-only, so the
    source file cannot be modified through the returned object; pickled
    objects (ops.npy, stat.npy) cannot be memory-mapped and are plain copies.
    """
    if allow_pickle:
        return np.load(path, allow_pickle=True)
    return np.load(path, mmap_mode="r")


def load_ops(plane_dir):
    """ops dict of a plane. Registration fields are also in reg_outputs.npy
    (suite2p >= 1.0), which is used to fill anything missing from ops.npy.
    """
    ops = {}
    ops_path = os.path.join(plane_dir, "ops.npy")
    if os.path.exists(ops_path):
        ops = load_npy(ops_path, allow_pickle=True).item()
    reg_path = os.path.join(plane_dir, "reg_outputs.npy")
    if os.path.exists(reg_path):
        reg = load_npy(reg_path, allow_pickle=True).item()
        for k, v in reg.items():
            ops.setdefault(k, v)
    if not ops:
        raise FileNotFoundError(f"no ops.npy or reg_outputs.npy in {plane_dir}")
    return ops


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
