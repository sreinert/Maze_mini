"""
Shared pieces of the abc_*.py QC scripts (ABC sequence-task data, Arya's
preprocessed layout; loaders in abc_data/): argument parsing, session loading,
figure saving, the phase/port colours, and the near-direct segment selection
(segment_route_stats / direct_only) shared by the decoding and tuning scripts.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from abc_data import find_sessions, load_session, maze
from abc_data.raw import find_raw_sessions
from suite2p_io import (INK_PRIMARY, INK_SECONDARY, SERIES_1, SERIES_2, SERIES_3, SURFACE,
                        default_out_dir)

EXAMPLE_ROOT = "/Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding"
# task phases / goal ports, in a fixed order
PHASE_NAMES = ("A→B", "B→C", "C→A")
GOAL_NAMES = ("A", "B", "C")
GOAL_COLORS = (SERIES_1, SERIES_2, SERIES_3)


def parser(doc, kind):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help=f"preprocessed ABC folder, e.g. {EXAMPLE_ROOT}")
    p.add_argument("--out", help=f"output folder (default ~/Desktop/maze_mini_qc/<name of root>/{kind})")
    p.add_argument("--session", action="append",
                   help="only sessions whose tag/label contains this (repeatable), e.g. 2026-08-13")
    return p


def setup(args, kind):
    """(root, out Path, [Session]) from parsed arguments; for a raw root
    (is_raw_root) the sessions are abc_data.raw.RawSession, which only the
    scripts that say so support (abc_sync_traces, abc_raw_qc, abc_raw_lag_qc)."""
    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        sys.exit(f"Folder not found: {root}")
    if is_raw_root(root):
        sessions = find_raw_sessions(root)
    else:
        sessions = [load_session(f) for f in find_sessions(root)]
    if args.session:
        sessions = [x for x in sessions if any(k in x.tag or k in x.label for k in args.session)]
        if not sessions:
            sys.exit(f"no session matches {args.session}")
    out = Path(os.path.expanduser(args.out or default_out_dir(root, kind)))
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(sessions)} session(s) in {root}")
    return root, out, sessions


def is_raw_root(root):
    """True for a folder of our own suite2p runs on raw data (abc_data/raw.py)
    rather than Arya's preprocessed layout."""
    return not os.path.isdir(os.path.join(root, "suite2p_output")) and bool(
        glob.glob(os.path.join(root, "funcimg", "*", "preprocessed", "suite2p", "plane0")))


def new_figure(*args, **kw):
    fig, axes = plt.subplots(*args, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def save(fig, path, title=None):
    if title:
        fig.suptitle(title, fontsize=11, color=INK_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def title(ax, text):
    ax.set_title(text, fontsize=10, color=INK_PRIMARY, loc="left")


def labels(ax, x=None, y=None):
    if x:
        ax.set_xlabel(x, fontsize=9, color=INK_SECONDARY)
    if y:
        ax.set_ylabel(y, fontsize=9, color=INK_SECONDARY)


def safe(label):
    return label.replace("/", "_")


def segment_route_stats(s, location):
    """Per frame, for the reward-to-reward segment it belongs to: excess node
    steps (distinct nodes visited, maze.node_path of location, minus the
    shortest path between its two goals) and wrong pokes (pokes at ports other
    than the two goals); NaN outside segments. Same segments as goal_progress
    (A/B/C_on from pyControl, consecutive in time)."""
    names = ("A_on", "B_on", "C_on")
    t = [s.pycontrol.state_times(n) for n in names]
    g = np.concatenate([np.full(len(v), i) for i, v in enumerate(t)])
    t = np.concatenate(t)
    o = np.argsort(t)
    to_frame = getattr(s, "pyc_to_aligned_frame", None) or s.pyc_to_frame
    k, g = to_frame(t[o]), g[o]
    ok = np.isfinite(k) & (k >= 0) & (k < s.n_frames)
    k, g = k[ok].astype(int), g[ok]
    pk = s.pycontrol.pokes()
    pf = to_frame(pk.time.to_numpy())
    pn = np.array([maze.port_to_node(p) for p in pk.port])
    D = maze.node_distances()
    location = np.asarray(location)
    ex, wp = np.full(s.n_frames, np.nan), np.full(s.n_frames, np.nan)
    for a, b, ga, gb in zip(k[:-1], k[1:], g[:-1], g[1:]):
        if b > a:
            src, dst = s.task[ga], s.task[gb]
            path = maze.node_path(location[a:b + 1])
            ex[a:b] = max(len(path) - 1, 0) - D[src, dst]
            wp[a:b] = np.sum((pf > a) & (pf < b) & ~np.isin(pn, [src, dst]))
    return ex, wp


def direct_only(gp, stats, max_excess=None, max_pokes=None):
    """Goal-progress labels set to -1 in segments with more than max_excess
    extra steps or more than max_pokes wrong pokes (stats from segment_route_stats)."""
    keep = np.ones(len(gp), bool)
    if max_excess is not None:
        keep &= stats[0] <= max_excess
    if max_pokes is not None:
        keep &= stats[1] <= max_pokes
    return np.where(keep, gp, -1)
