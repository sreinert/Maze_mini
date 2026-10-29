"""
Registration quality control for every suite2p session below a preprocessed folder.

    uv run python quality_control/registration_qc.py <path> [--out DIR]

<path> can be anything from a folder of many sessions down to a single plane
folder, e.g.

    /Volumes/behrens/Sandra/mini2p/fscan                 (all tests)
    /Volumes/behrens/Sandra/mini2p/fscan/test4           (one two-plane test)

Only registration output is needed (ops.npy or reg_outputs.npy), so this also
works on sessions where suite2p stopped before detection/extraction.

One figure per session (the folder holding suite2p/), with one row per plane:

    - registered mean image of the functional channel (meanImg) and of the other,
      structural channel (meanImg_chan2), contrast-clipped at the --clip
      percentiles; the dashed box is the region suite2p keeps after
      registration (yrange/xrange), outside it pixels were shifted in from the edge
    - rigid offsets per frame (yoff, xoff, in pixels)
    - phase-correlation peak per frame (corrXY), with frames suite2p flagged as
      bad (badframes) marked below the trace; dips mean frames that matched the
      reference poorly
    - for non-rigid registration, the largest block shift per frame

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of path>/registration_qc):

    <session label>.png   one figure per session
    summary.csv           per-plane registration metrics

Inputs are opened read-only and nothing is written next to the data.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite2p_io import (GRIDLINE, INK_PRIMARY, INK_SECONDARY, SERIES_1, SERIES_2, SURFACE,
                        default_out_dir, find_plane_dirs, load_ops, plane_label,
                        session_label, style_axis)


def _align_by_chan2(ops):
    reg = ops.get("registration")
    if isinstance(reg, dict) and "align_by_chan2" in reg:
        return bool(reg["align_by_chan2"])
    return bool(ops.get("align_by_chan2", False))


def _channels(ops):
    """(functional, other) raw channel numbers; other is None for one channel."""
    func = int(ops.get("functional_chan", 1))
    if int(ops.get("nchannels", 1)) < 2:
        return func, None
    return func, 2 if func == 1 else 1


def plane_metrics(ops, label):
    yoff = np.asarray(ops.get("yoff", []), dtype=float)
    xoff = np.asarray(ops.get("xoff", []), dtype=float)
    corr = np.asarray(ops.get("corrXY", []), dtype=float)
    bad = np.asarray(ops.get("badframes", np.zeros(len(yoff), bool)), dtype=bool)
    m = {
        "plane": label,
        "fs": ops.get("fs"),
        "n_frames": len(yoff),
        "align_by_chan2": _align_by_chan2(ops),
        "max_abs_yoff": float(np.max(np.abs(yoff))) if len(yoff) else np.nan,
        "max_abs_xoff": float(np.max(np.abs(xoff))) if len(xoff) else np.nan,
        "sd_yoff": float(np.std(yoff)) if len(yoff) else np.nan,
        "sd_xoff": float(np.std(xoff)) if len(xoff) else np.nan,
        "median_corrXY": float(np.median(corr)) if len(corr) else np.nan,
        "min_corrXY": float(np.min(corr)) if len(corr) else np.nan,
        "n_badframes": int(bad.sum()),
        "frac_badframes": float(bad.mean()) if len(bad) else np.nan,
    }
    if "xoff1" in ops and "yoff1" in ops:
        nr = np.hypot(np.asarray(ops["yoff1"], float), np.asarray(ops["xoff1"], float))
        m["max_nonrigid_shift"] = float(nr.max())
    return m


def _show_mean_image(ax, img, title, clip, ops):
    ax.set_title(title, fontsize=9.5, color=INK_PRIMARY, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if img is None:
        ax.set_facecolor(SURFACE)
        ax.text(0.5, 0.5, "not saved\n(single channel?)", ha="center", va="center",
                fontsize=9, color=INK_SECONDARY, transform=ax.transAxes)
        return
    img = np.asarray(img, dtype=float)
    lo, hi = np.percentile(img, clip)
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    yr, xr = ops.get("yrange"), ops.get("xrange")
    if yr is not None and xr is not None and len(yr) == 2 and len(xr) == 2:
        ax.add_patch(Rectangle((xr[0] - 0.5, yr[0] - 0.5), xr[1] - xr[0], yr[1] - yr[0],
                               fill=False, edgecolor=SERIES_2, linewidth=1, linestyle="--"))


def _plot_plane(subfig, ops, label, clip):
    fs = ops.get("fs") or 1.0
    yoff = np.asarray(ops.get("yoff", []), dtype=float)
    xoff = np.asarray(ops.get("xoff", []), dtype=float)
    corr = np.asarray(ops.get("corrXY", []), dtype=float)
    bad = np.asarray(ops.get("badframes", np.zeros(len(yoff), bool)), dtype=bool)
    t = np.arange(len(yoff)) / fs
    nonrigid = "xoff1" in ops and "yoff1" in ops
    n_rows = 3 if nonrigid else 2

    gs = subfig.add_gridspec(n_rows, 4, width_ratios=[1, 1, 1.5, 1.5], hspace=0.02, wspace=0.03)
    func, other = _channels(ops)
    by2 = _align_by_chan2(ops)
    _show_mean_image(subfig.add_subplot(gs[:, 0]), ops.get("meanImg"),
                     f"functional (ch{func}), meanImg" + ("" if by2 else ", reg. target"), clip, ops)
    _show_mean_image(subfig.add_subplot(gs[:, 1]), ops.get("meanImg_chan2"),
                     (f"structural (ch{other}), meanImg_chan2" + (", reg. target" if by2 else ""))
                     if other else "second channel", clip, ops)

    ax = subfig.add_subplot(gs[0, 2:])
    style_axis(ax)
    ax.plot(t, yoff, color=SERIES_1, linewidth=0.9, label="y")
    ax.plot(t, xoff, color=SERIES_2, linewidth=0.9, label="x")
    ax.set_ylabel("rigid shift (px)", fontsize=9, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, ncol=2, loc="upper right")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_xlim(t[0] if len(t) else 0, t[-1] if len(t) else 1)
    axes = [ax]

    ax = subfig.add_subplot(gs[1, 2:], sharex=axes[0])
    style_axis(ax)
    ax.plot(t, corr, color=SERIES_1, linewidth=0.9)
    if bad.any():
        floor = np.nanmin(corr) - 0.05 * (np.nanmax(corr) - np.nanmin(corr) + 1e-9)
        ax.plot(t[bad], np.full(bad.sum(), floor), "|", color=INK_PRIMARY, markersize=8,
                label=f"bad frames ({bad.sum()})")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower right")
    ax.set_ylabel("phase corr. peak\n(corrXY)", fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    axes.append(ax)

    if nonrigid:
        ax = subfig.add_subplot(gs[2, 2:], sharex=axes[0])
        style_axis(ax)
        nr = np.hypot(np.asarray(ops["yoff1"], float), np.asarray(ops["xoff1"], float))
        ax.plot(t, nr.max(axis=1), color=SERIES_1, linewidth=0.9)
        ax.set_ylabel("max non-rigid\nblock shift (px)", fontsize=9, color=INK_SECONDARY)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        axes.append(ax)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("time (s)", fontsize=9, color=INK_SECONDARY)

    m = plane_metrics(ops, label)
    subfig.suptitle(
        f"{label}   |   {fs:g} Hz, {m['n_frames']} frames   |   "
        f"max |shift| y {m['max_abs_yoff']:.0f} px, x {m['max_abs_xoff']:.0f} px   |   "
        f"median corrXY {m['median_corrXY']:.3g}   |   "
        f"bad frames {m['n_badframes']} ({m['frac_badframes']:.1%})",
        fontsize=10, color=INK_PRIMARY, x=0.01, ha="left")
    subfig.set_facecolor(SURFACE)
    return m


def plot_session(planes, root, out_path, clip):
    """planes: list of (plane_dir, ops) for one session, in plane order."""
    fig = plt.figure(figsize=(18, 4.2 * len(planes)), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    subfigs = fig.subfigures(len(planes), 1, squeeze=False)[:, 0]
    rows = [_plot_plane(sf, ops, plane_label(d, root), clip) for sf, (d, ops) in zip(subfigs, planes)]
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="preprocessed folder (any level down to a suite2p plane folder)")
    p.add_argument("--out", help="output folder (default ~/Desktop/maze_mini_qc/<name of path>/registration_qc)")
    p.add_argument("--clip", type=float, nargs=2, default=(1, 99.5), metavar=("LO", "HI"),
                   help="mean-image percentiles mapped to black/white (default 1 99.5)")
    a = p.parse_args()

    root = os.path.abspath(os.path.expanduser(a.path))
    if not os.path.isdir(root):
        sys.exit(f"Folder not found: {root}")
    out = Path(os.path.expanduser(a.out or default_out_dir(root, "registration_qc")))
    out.mkdir(parents=True, exist_ok=True)

    plane_dirs = find_plane_dirs(root, required=("ops.npy", "reg_outputs.npy"))
    if not plane_dirs:
        sys.exit(f"No suite2p plane folders with ops.npy/reg_outputs.npy found below {root}")

    sessions = defaultdict(list)
    for d in plane_dirs:
        try:
            ops = load_ops(d)
        except Exception as e:
            print(f"[warn] skipping {plane_label(d, root)}: {e}")
            continue
        if "yoff" not in ops:
            print(f"[warn] skipping {plane_label(d, root)}: no registration output in ops")
            continue
        sessions[session_label(d, root)].append((d, ops))
    print(f"Found {sum(len(v) for v in sessions.values())} registered plane(s) in "
          f"{len(sessions)} session(s) below {root}")

    rows = []
    for label, planes in sessions.items():
        out_path = out / (label.replace("/", "_") + ".png")
        session_rows = plot_session(planes, root, out_path, a.clip)
        rows.extend(session_rows)
        for m in session_rows:
            print(f"{m['plane']}: max |shift| y {m['max_abs_yoff']:.0f} / x {m['max_abs_xoff']:.0f} px, "
                  f"median corrXY {m['median_corrXY']:.3g}, {m['n_badframes']} bad frames")

    if rows:
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(f"Saved figures and table to {out}")


if __name__ == "__main__":
    main()
