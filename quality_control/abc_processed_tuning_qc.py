"""
Goal-progress tuning on Arya's processed sessions, run as much like the raw
sessions as possible (abc_raw_tuning_qc), at one or several lags between
activity and behaviour.

    uv run python quality_control/abc_processed_tuning_qc.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding
    uv run python quality_control/abc_processed_tuning_qc.py <root> --lags 0 -250 -490 -500 -510

Same pipeline as the raw sessions: abc_data.raw.ProcessedSession gives the
processed sessions the RawSession interface, i.e. suite2p F - 0.7 * Fneu of
suite2p_output/<tag>_... (all ROIs; iscell is all 1 there), background regressor
= mean Fneu over ROIs, suite2p frames timed by the DAQ FrameClock, goal progress
from the pyControl log through the rsync fit. Arya's aligned arrays (ROICaT
activity, trial_times, XY) are not used. All analysis and figure code is
abc_raw_tuning_qc's (analyse, run_session, plots), with the same defaults
(10 bins, background regression, no smoothing; polar display 20 bins, sigma 2).

Lag convention (as in abc_raw_tuning_qc --lag): behaviour at frame t is paired
with activity at frame t + lag. The ~500-frame offset in the processed data
(CLAUDE.md, known findings: activity ~500 frames *earlier* than its behaviour)
is therefore lag = -500. Default --lags 0 -500.

Near-direct segments: --max-excess-steps / --max-wrong-pokes as in
abc_raw_tuning_qc (routes from --tracking if present, else Arya's maze_location
moved to suite2p frames).

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of root>/tuning_qc):
    tuning/<session>_<regress>_lag<L>_...        heatmaps, polar plots, segment
                                                 tests per lag (as for raw)
    tuning/<session>_lag_sorting.png             cross-lag sorting grid: rows =
                                                 lag used to select (p < alpha)
                                                 and sort cells by preferred bin,
                                                 columns = lag whose (all-trial)
                                                 curves are shown
    lags_summary.png                             tuned fraction, odd/even
                                                 preferred-bin agreement and
                                                 segment tests per session vs lag
    summary_<...>_lags<...>.csv, cells_<...>_lags<...>.csv
"""
import numpy as np
import pandas as pd

from abc_common import labels, new_figure, parser, safe, save, title
from abc_data.raw import find_processed_sessions
from abc_raw_tuning_qc import _norm, add_tuning_args, filter_tag, run_session, segment_mask
from suite2p_io import GRIDLINE, INK_MUTED, colormap, default_out_dir, style_axis

import os
import sys
from pathlib import Path


def plot_lag_sorting(s, results, lags, n_bins, alpha, out_path):
    """Grid: row i = cells significant at lags[i], sorted by their preferred bin at
    lags[i]; column j = the same cells' tuning curves at lags[j]."""
    n = len(lags)
    fig, axes = new_figure(n, n, figsize=(5.2 * n + 1.5, 4.4 * n + 0.8), squeeze=False)
    cmap = colormap()
    im = None
    for i, li in enumerate(lags):
        ri = results[li]
        sig = np.flatnonzero(ri["p"] < alpha)
        order = sig[np.argsort(ri["pref"][sig], kind="stable")]
        for j, lj in enumerate(lags):
            ax = axes[i, j]
            style_axis(ax)
            if len(order):
                im = ax.imshow(_norm(results[lj]["T"])[:, order].T, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                               interpolation="nearest", extent=[0, 1, len(order), 0])
            rj = results[lj]
            agree = np.nan
            if len(order):
                d = np.abs(ri["pref"][order] - rj["pref"][order])
                agree = float(np.mean(np.minimum(d, n_bins - d) <= 1))
            same = " (same lag)" if li == lj else ""
            title(ax, f"sorted at lag {li} → shown at lag {lj}\n{len(order)} cells, pref. bin ±1: {agree:.0%}{same}")
            labels(ax, "goal progress" if i == n - 1 else None, "cell" if j == 0 else None)
    fig.suptitle(f"{s.label}  |  goal-progress tuning sorted at one lag and shown at another  |  "
                 f"lag = activity frame minus behaviour frame", fontsize=11, x=0.01, ha="left")
    fig.subplots_adjust(left=0.07, right=0.86, top=0.88, bottom=0.08, hspace=0.45, wspace=0.25)
    if im is not None:
        cax = fig.add_axes([0.89, 0.3, 0.015, 0.4])
        fig.colorbar(im, cax=cax, label="normalised activity (min-max per cell)")
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_lags_summary(summary, lags, alpha, n_bins, out_path, root):
    sessions = list(dict.fromkeys(summary.session))
    fig, axes = new_figure(1, 4, figsize=(22, 5.2))
    ramp = colormap()
    cols = (("frac_significant", "goal-progress tuned cells", alpha),
            ("frac_pref_within1_sig", "tuned cells: preferred bin odd/even within ±1", 3 / n_bins),
            ("frac_seg_consistent_all", "all cells: segment-consistent", alpha),
            ("frac_pref_match_all", "all cells: preferred phase matches across segments", alpha))
    for ax, (col, ttl, chance) in zip(axes, cols):
        style_axis(ax)
        for k, s in enumerate(sessions):
            d = summary[summary.session == s].set_index("lag").reindex(lags)
            ax.plot(range(len(lags)), d[col], "o-", color=ramp(0.3 + 0.7 * k / max(len(sessions) - 1, 1)),
                    markersize=6, linewidth=1.3, label=s.split("/")[-1])
        ax.axhline(chance, color=INK_MUTED, linestyle="--", linewidth=1, label="chance")
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels([str(lg) for lg in lags])
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        title(ax, ttl)
        labels(ax, "lag (activity frame − behaviour frame)", "fraction")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    save(fig, out_path, f"Goal-progress tuning vs lag, processed sessions (raw-like pipeline)  |  {root}")


def main():
    p = add_tuning_args(parser(__doc__, "tuning_qc"))
    p.add_argument("--lags", type=int, nargs="+", default=[0, -500],
                   help="lags to run: behaviour at t paired with activity at t + lag (default 0 -500)")
    a = p.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root))
    if not os.path.isdir(root):
        sys.exit(f"Folder not found: {root}")
    sessions = find_processed_sessions(root)
    if a.session:
        sessions = [x for x in sessions if any(k in x.tag or k in x.label for k in a.session)]
    if not sessions:
        sys.exit("no sessions")
    out = Path(os.path.expanduser(a.out or default_out_dir(root, "tuning_qc")))
    (out / "tuning").mkdir(parents=True, exist_ok=True)
    print(f"{len(sessions)} session(s) in {root}; lags {a.lags}")
    rows, cells = [], []
    for s in sessions:
        results = {}
        keep = segment_mask(s, a, root)
        for lag in a.lags:
            row, c, r = run_session(s, a, lag, out, keep)
            rows.append(row)
            cells.append(c)
            results[lag] = r
        if len(a.lags) > 1:
            plot_lag_sorting(s, results, a.lags, a.bins, a.alpha,
                             out / "tuning" / f"{safe(s.label)}_lag_sorting{filter_tag(a)}.png")
    sfx = f"{a.regress}_bins{a.bins}_smooth{a.smooth:g}_lags{'_'.join(map(str, a.lags))}{filter_tag(a)}"
    summary = pd.DataFrame(rows)
    summary.to_csv(out / f"summary_{sfx}.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(out / f"cells_{sfx}.csv", index=False)
    plot_lags_summary(summary, a.lags, a.alpha, a.bins, out / f"lags_summary{filter_tag(a)}.png", root)
    with pd.option_context("display.width", 200):
        print(summary.pivot(index="session", columns="lag", values="frac_significant").round(2).to_string())
    print(f"Saved figures and tables to {out}")


if __name__ == "__main__":
    main()
