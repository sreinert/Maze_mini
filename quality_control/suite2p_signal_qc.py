"""
Fluorescence traces and signal-quality metrics for every suite2p plane below a
preprocessed folder.

    uv run python quality_control/suite2p_signal_qc.py <path> [--out DIR] [options]

<path> can be anything from a folder of many sessions down to a single plane
folder, e.g.

    /Volumes/behrens/Sandra/mini2p/fscan                                 (all tests)
    /Volumes/behrens/Sandra/mini2p/fscan/test2                           (one test)
    /Volumes/behrens/Sandra/mini2p/fscan/test2/preprocessed/suite2p/plane0

Every plane folder with F.npy below it is processed (suite2p_io.find_plane_dirs).
The metrics and the trace plots follow Bonsai_ABCD's compare_signal_quality.py
and plot_high_snr_traces.py, so values can be compared directly with the ABCD
sessions. The differences: this script takes a path instead of a YAML settings
file, works per plane, and sets the trace window in seconds because Mini2P
recordings differ in frame rate (20-35 Hz in the fscan tests) and length.

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of path>/signal_qc):

    traces/<label>.png  per plane: neuropil-corrected dF/F (F - --trace-neucoeff *
                        Fneu, default 0.7, normalised by the raw-F baseline; see
                        signal_quality_metrics) of the top --n-cells cells by SNR next
                        to the same number of random cells, under the population
                        mean (grey) to show how much of each trace is shared.
                        All traces share one dF/F scale (scale bar) set by the
                        typical cell; a much larger cell is shrunk by a round
                        factor, marked in its label as "shown /N"
    summary.png         per-plane metric distributions (one column per plane)
    summary.csv         per-plane summary metrics
    per_cell.csv        per-cell metrics for every plane

The trace figures and the summary metrics are computed separately: traces use
--trace-neucoeff (default 0.7), the metrics --neucoeff (default 0 = raw F, as in
Bonsai_ABCD, so the numbers stay comparable with the ABCD sessions).

Inputs are opened read-only and nothing is written next to the data.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signal_quality_metrics import (compute_channel2_metrics, load_percell_traces,
                                    select_cells, summarise_traces)
from suite2p_io import (INK_MUTED, INK_PRIMARY, INK_SECONDARY, GRIDLINE, SERIES_1, SERIES_2,
                        SURFACE, default_out_dir, find_plane_dirs, plane_label, style_axis)

LABEL_BOX = dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.85)


# ---------------------------------------------------------------- traces

def _nice_bar(span):
    """Largest 'round' dF/F value (1, 2, 5 x 10^k) not above span."""
    k = np.floor(np.log10(span))
    for m in (5, 2, 1):
        if m * 10 ** k <= span:
            return m * 10 ** k
    return 10 ** k


def _trace_ranges(traces, idx, start, end):
    """0.5-99.5 percentile range of each shown cell's dF/F in the window."""
    if len(idx) == 0:
        return np.array([])
    seg = traces["dff"][idx, start:end]
    return np.nanpercentile(seg, 99.5, axis=1) - np.nanpercentile(seg, 0.5, axis=1)


def _trace_offset(ranges):
    """Spacing between stacked traces, from the typical (median) shown cell so a
    single very large cell does not flatten all the others."""
    r = ranges[np.isfinite(ranges) & (ranges > 0)]
    return float(np.median(r) * 1.3) if len(r) else 1.0


def _shrink_factor(rng, offset):
    """1 for cells that fit their row; otherwise the smallest 'round' factor
    (2, 5, 10, 20, ...) that makes them fit. Drawn traces are divided by it."""
    if not np.isfinite(rng) or rng <= 1.5 * offset:
        return 1
    k = 0
    while True:
        for m in (2, 5, 10):
            f = m * 10 ** k
            if rng / f <= 1.5 * offset:
                return f
        k += 1


def _trace_panel(ax, traces, idx, start, end, fs, title, offset):
    t = np.arange(start, end) / fs
    pop = traces["pop_mean"][start:end]
    base = len(idx) * offset
    ax.plot(t, pop - np.nanmedian(pop) + base, color=INK_MUTED, linewidth=0.8)
    ax.text(t[0], base + 0.45 * offset, "population mean", fontsize=7.5, color=INK_SECONDARY,
            ha="left", va="bottom", zorder=5, bbox=LABEL_BOX)
    ranges = _trace_ranges(traces, idx, start, end)
    for i, c in enumerate(idx):
        trace = traces["dff"][c, start:end]
        f = _shrink_factor(ranges[i], offset)
        ax.plot(t, (trace - np.nanmedian(trace)) / f + i * offset, color=SERIES_1, linewidth=0.6)
        shrink = f"  (shown \u00f7{f})" if f > 1 else ""
        ax.text(t[0], (i + 0.45) * offset,
                f"ROI {traces['cell_roi_idx'][c]}  SNR={traces['snr'][c]:.1f}  "
                f"r_pop={traces['pop_corr'][c]:.2f}{shrink}",
                fontsize=7.5, color=INK_SECONDARY, ha="left", va="bottom", zorder=5, bbox=LABEL_BOX)

    # vertical scale bar in the empty band below the lowest trace
    bar = _nice_bar(0.6 * offset)
    x_bar = t[-1] - 0.01 * (t[-1] - t[0])
    y0 = -0.9 * offset
    ax.plot([x_bar, x_bar], [y0, y0 + bar], color=INK_PRIMARY, linewidth=1.5, solid_capstyle="butt")
    ax.text(x_bar - 0.005 * (t[-1] - t[0]), y0 + bar / 2, f"{bar * 100:g}% dF/F",
            fontsize=8, color=INK_PRIMARY, ha="right", va="center")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-offset, (len(idx) + 1.2) * offset)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (s)", fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left")


def plot_traces(traces, label, out_path, n_cells, window_s, snr_threshold, neucoeff):
    """traces: load_percell_traces() output computed with the neuropil
    coefficient used for plotting (neucoeff)."""
    fs = traces["fs"] if traces["fs"] else 1.0
    n_frames = traces["n_frames"]
    win = n_frames if window_s <= 0 else min(n_frames, int(round(window_s * fs)))
    start = (n_frames - win) // 2
    end = start + win

    top = select_cells(traces, n_cells, order="snr")
    rand = select_cells(traces, n_cells, order="random")
    n_show = max(len(top), 1)
    # one dF/F scale for both panels
    offset = _trace_offset(np.concatenate([_trace_ranges(traces, top, start, end),
                                           _trace_ranges(traces, rand, start, end)]))

    fig, axes = plt.subplots(1, 2, figsize=(17, max(4, 0.75 * (n_show + 1) + 1.5)))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style_axis(ax)
    _trace_panel(axes[0], traces, top, start, end, fs, f"Top {len(top)} cells by SNR", offset)
    _trace_panel(axes[1], traces, rand, start, end, fs, f"{len(rand)} random cells", offset)

    snr = traces["snr"]
    n_total = len(snr)
    n_pass = int(np.sum(snr > snr_threshold))
    n_nan = int(np.sum(~np.isfinite(snr)))
    nan_note = f", {n_nan} without a positive raw baseline (not shown)" if n_nan else ""
    trace_note = (f"dF/F = (Fc - F0c) / F0, Fc = F - {neucoeff:g}*Fneu" if neucoeff else "dF/F of raw F")
    fig.suptitle(
        f"{label}   |   {trace_note}   |   {n_pass}/{n_total} cells with SNR > {snr_threshold}{nan_note}   |   "
        f"median r_pop {np.nanmedian(traces['pop_corr']):.2f}   |   "
        f"{start / fs:.0f}-{end / fs:.0f} s of {n_frames / fs:.0f} s at {fs:g} Hz",
        fontsize=10, color=INK_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- summary

def _cell_panel(ax, per_cell, labels, metric, title, ylabel, log=False, threshold=None):
    """One column per plane: every cell as a faint dot, the median as a dark bar."""
    rng = np.random.default_rng(0)
    for i, lab in enumerate(labels):
        v = per_cell.loc[per_cell["plane"] == lab, metric].to_numpy(dtype=float)
        v = v[np.isfinite(v) & ((v > 0) if log else True)]
        if len(v) == 0:
            continue
        few = len(v) < 100  # small test FOVs: make each cell visible
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=12 if few else 4, color=SERIES_1,
                   alpha=0.7 if few else 0.3, edgecolor="none", zorder=2)
        med = np.median(v)
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK_PRIMARY, linewidth=2, zorder=3)
    if threshold is not None:
        ax.axhline(threshold, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    if log:
        ax.set_yscale("log")
    _finish_panel(ax, labels, title, ylabel)


def _scalar_panel(ax, summary, labels, series, title, ylabel):
    """One dot per plane for each (column, legend label, colour) in series."""
    for col, name, color in series:
        if col not in summary:
            continue
        y = summary.set_index("plane").reindex(labels)[col].to_numpy(dtype=float)
        ax.plot(range(len(labels)), y, "o", color=color, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    _finish_panel(ax, labels, title, ylabel)


def _finish_panel(ax, labels, title, ylabel):
    style_axis(ax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_title(title, fontsize=10.5, color=INK_PRIMARY, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)


def plot_summary(summary, per_cell, out_path, title, params):
    labels = list(summary["plane"])
    width = max(20, 5 * 0.3 * len(labels) + 5)
    fig, axes = plt.subplots(2, 5, figsize=(width, 10))
    fig.patch.set_facecolor(SURFACE)
    a = axes.ravel()

    _cell_panel(a[0], per_cell, labels, "F0", "Baseline fluorescence per cell",
                "F0 (a.u., 8th pct. of F)", log=True)
    _scalar_panel(a[1], summary, labels,
                  [("mean_img_mean", "functional", SERIES_1),
                   ("fov_chan2_mean", "structural (chan2)", SERIES_2)],
                  "Mean-image brightness", "mean pixel intensity (a.u.)")
    _scalar_panel(a[2], summary, labels,
                  [("n_cells", "iscell", SERIES_1), ("n_roi_total", "all ROIs", SERIES_2)],
                  "Number of ROIs", "count")
    _cell_panel(a[3], per_cell, labels, "snr", "SNR per cell", "peak dF/F (p99) / noise SD",
                log=True, threshold=params["snr_threshold"])
    _cell_panel(a[4], per_cell, labels, "skew", "dF/F skewness per cell", "skew",
                threshold=params["skew_threshold"])
    _scalar_panel(a[5], summary, labels,
                  [("frac_cells_snr_gt_thresh", f"SNR > {params['snr_threshold']}", SERIES_1),
                   ("frac_cells_skew_gt_thresh", f"skew > {params['skew_threshold']}", SERIES_2)],
                  "Cells with transient-like signal", "fraction of cells")
    _cell_panel(a[6], per_cell, labels, "event_frac", "Event-like frames per cell",
                f"fraction of frames > {params['event_sigma']} noise SD")
    _cell_panel(a[7], per_cell, labels, "neu_corr", "Cell-neuropil correlation",
                "corr(raw F, Fneu)")
    _cell_panel(a[8], per_cell, labels, "noise_sd", "Noise per cell",
                "noise SD (dF/F units)", log=True)
    _cell_panel(a[9], per_cell, labels, "pop_corr", "Correlation with population mean",
                "corr(dF/F, mean dF/F of all cells)")

    fig.suptitle(title, fontsize=11, color=INK_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- main

PER_CELL_COLS = ("F0", "noise_sd", "peak_dff", "skew", "snr", "event_frac", "neu_corr", "pop_corr")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="preprocessed folder (any level down to a suite2p plane folder)")
    p.add_argument("--out", help="output folder (default ~/Desktop/maze_mini_qc/<name of path>/signal_qc)")
    p.add_argument("--n-cells", type=int, default=10, help="cells per trace panel (default 10)")
    p.add_argument("--window", type=float, default=300,
                   help="trace window in s, centred on the recording; 0 = whole recording (default 300)")
    p.add_argument("--trace-neucoeff", type=float, default=0.7,
                   help="trace figures: plot dF/F of F - trace_neucoeff * Fneu (default 0.7)")
    p.add_argument("--neucoeff", type=float, default=0,
                   help="summary metrics: subtract neucoeff * Fneu before dF/F; "
                        "0 = raw F as in Bonsai_ABCD (default 0)")
    p.add_argument("--baseline-percentile", type=float, default=8)
    p.add_argument("--event-sigma", type=float, default=3)
    p.add_argument("--snr-threshold", type=float, default=5)
    p.add_argument("--skew-threshold", type=float, default=1)
    p.add_argument("--no-traces", action="store_true", help="only the summary figure and tables")
    a = p.parse_args()

    root = os.path.abspath(os.path.expanduser(a.path))
    if not os.path.isdir(root):
        sys.exit(f"Folder not found: {root}")
    out = Path(os.path.expanduser(a.out or default_out_dir(root, "signal_qc")))
    out.mkdir(parents=True, exist_ok=True)
    params = {"baseline_percentile": a.baseline_percentile, "event_sigma": a.event_sigma,
              "snr_threshold": a.snr_threshold, "skew_threshold": a.skew_threshold}

    plane_dirs = find_plane_dirs(root)
    if not plane_dirs:
        sys.exit(f"No suite2p plane folders with F.npy found below {root}")
    print(f"Found {len(plane_dirs)} suite2p plane folder(s) below {root}")
    if not a.no_traces:
        (out / "traces").mkdir(exist_ok=True)

    rows, cell_tables = [], []
    for plane_dir in plane_dirs:
        label = plane_label(plane_dir, root)
        traces = load_percell_traces(plane_dir, a.baseline_percentile, a.event_sigma, a.neucoeff)
        if "error" in traces:
            print(f"[warn] skipping {label}: {traces['error']}")
            continue
        metrics = summarise_traces(traces, a.snr_threshold, a.skew_threshold)
        chan2 = compute_channel2_metrics(plane_dir, ops=traces["ops"])
        if "error" in chan2:
            print(f"[warn] no channel-2 metrics for {label}: {chan2['error']}")
        else:
            metrics.update(chan2)
        rows.append({"plane": label, **metrics})

        cells = pd.DataFrame({c: traces[c] for c in PER_CELL_COLS})
        cells.insert(0, "roi", traces["cell_roi_idx"])
        cells.insert(0, "plane", label)
        cell_tables.append(cells)

        print(f"{label}: {metrics['n_cells']} cells, median SNR {metrics['median_snr']:.1f}, "
              f"{metrics['frac_cells_snr_gt_thresh']:.0%} with SNR > {a.snr_threshold}, "
              f"median r_pop {metrics['median_pop_corr']:.2f}")
        if not a.no_traces:
            plot_tr = traces if a.trace_neucoeff == a.neucoeff else load_percell_traces(
                plane_dir, a.baseline_percentile, a.event_sigma, a.trace_neucoeff)
            plot_traces(plot_tr, label, out / "traces" / (label.replace("/", "_") + ".png"),
                        a.n_cells, a.window, a.snr_threshold, a.trace_neucoeff)

    if not rows:
        sys.exit("No plane could be loaded.")
    summary = pd.DataFrame(rows)
    if "fov_chan2_mean" in summary:
        summary["chan1_chan2_ratio"] = summary["mean_img_mean"] / summary["fov_chan2_mean"]
    per_cell = pd.concat(cell_tables, ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    per_cell.to_csv(out / "per_cell.csv", index=False)
    plot_summary(summary, per_cell, out / "summary.png",
                 f"Signal quality ({'F - %g*Fneu' % a.neucoeff if a.neucoeff else 'raw F'})  |  {root}", params)
    print(f"Saved figures and tables to {out}")


if __name__ == "__main__":
    main()
