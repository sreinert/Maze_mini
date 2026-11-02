"""
Does the neural activity carry task and position information? Event-locked
activity at the goal pokes and spatial tuning on the maze, each tested for
reproducibility across trials against a circular-shift null.

    uv run python quality_control/abc_neural_task_qc.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding

Activity = the aligned ROICaT activity (cells present in the session), z-scored
per cell: it is in arbitrary F units (baseline-subtracted Fcorr, not dF/F; see
abc_signal_qc), so amplitudes are only comparable after z-scoring.

    PETH          mean activity from -2 to +3 s around A_on, B_on, C_on.
                  Reliability per cell and goal = correlation of the PETHs of
                  odd vs even trials.
    spatial map   mean activity per maze location (nodes and edges with enough
                  frames). Reliability = correlation of the maps from odd vs even
                  trials.
    null          the same reliabilities after circularly shifting each cell's
                  activity by a random amount (>= 30 s) relative to behaviour,
                  --n-shuffles times; a cell is 'reliable' if it beats the 95th
                  percentile of its own null. ~5 % of cells pass by chance.

--lag N pairs behaviour at t with activity at t - N frames. At lag 0 the AB_08
sessions are at chance; at --lag 490 (~65 s, where decoding peaks, see
abc_decoding_qc) 10-16 % of cells have reliable PETHs in 5 of 6 sessions.

Heatmaps are cross-validated: cells are ordered by their peak in the odd
trials and the even-trial maps are shown, so structure that survives is not
an artefact of sorting noise.

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of root>/neural_task_qc):
    sessions/<session>.png   PETH heatmaps (A, B, C), spatial map heatmap,
                             reliability vs null
    summary.png              fraction of reliable cells per session and measure
    neural_task.csv          per session; reliability.csv per cell
"""
import numpy as np
import pandas as pd

from abc_common import GOAL_COLORS, GOAL_NAMES, labels, new_figure, parser, safe, save, setup, title
from abc_data import maze
from suite2p_io import GRIDLINE, INK_MUTED, INK_PRIMARY, SERIES_1, colormap, style_axis

PRE_S, POST_S = 2.0, 3.0
MIN_LOC_FRAMES = 20   # per half; locations visited less are left out of the maps


def zscored_activity(s, lag=0):
    """z-scored activity; lag > 0 pairs behaviour at t with activity at t - lag
    (frames without a partner are set to 0 = the mean)."""
    x = np.asarray(s.activity, dtype=np.float64)[:, s.present]
    x = np.nan_to_num((x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0))
    if lag:
        x = np.roll(x, lag, axis=0)
        if lag > 0:
            x[:lag] = 0
        else:
            x[lag:] = 0
    return x


def goal_events(s):
    """Aligned frames of A_on (excluding the first trial's 0), B_on, C_on."""
    tt = s.trial_times
    return [tt[1:, 0], tt[:, 1], tt[:, 2]]


def peth(x, frames, pre, post):
    """(n_events, pre+post, n_cells), events too close to the edges dropped."""
    frames = frames[(frames - pre >= 0) & (frames + post <= len(x))]
    return np.stack([x[f - pre:f + post] for f in frames]) if len(frames) else np.zeros((0, pre + post, x.shape[1]))


def _rowcorr(a, b):
    """Correlation per cell of a and b, both (n_bins, n_cells)."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def trial_index(s):
    """(T,) trial number of each aligned frame (-1 before the first A)."""
    idx = np.full(s.n_frames, -1)
    for i, (a, *_, end) in enumerate(s.trial_times):
        idx[a:end] = i
    return idx


def peth_reliability(x, events, pre, post):
    out = []
    for ev in events:
        P = peth(x, ev, pre, post)
        if len(P) < 4:
            out.append(np.full(x.shape[1], np.nan))
            continue
        out.append(_rowcorr(P[0::2].mean(0), P[1::2].mean(0)))
    return np.array(out)  # (3, n_cells)


def spatial_maps(x, location, half, locs):
    """(2, n_locs, n_cells) mean activity per location in the two halves."""
    M = np.full((2, len(locs), x.shape[1]), np.nan)
    for h in (0, 1):
        for j, l in enumerate(locs):
            m = (location == l) & (half == h)
            if m.any():
                M[h, j] = x[m].mean(0)
    return M


def analyse(s, n_shuffles, lag=0, seed=0):
    x = zscored_activity(s, lag)
    pre, post = int(round(PRE_S * s.fs_measured)), int(round(POST_S * s.fs_measured))
    events = goal_events(s)
    half = trial_index(s) % 2
    half[trial_index(s) < 0] = -1
    locs = [l for l in maze.LOCATIONS
            if all(np.sum((s.location == l) & (half == h)) >= MIN_LOC_FRAMES for h in (0, 1))]

    rel_peth = peth_reliability(x, events, pre, post)
    M = spatial_maps(x, s.location, half, locs)
    rel_space = _rowcorr(M[0], M[1])

    rng = np.random.default_rng(seed)
    lo = int(30 * s.fs_measured)
    null_peth = np.zeros((n_shuffles,) + rel_peth.shape)
    null_space = np.zeros((n_shuffles, x.shape[1]))
    for k in range(n_shuffles):
        shifts = rng.integers(lo, len(x) - lo, x.shape[1])
        xs = np.stack([np.roll(x[:, i], shifts[i]) for i in range(x.shape[1])], axis=1)
        null_peth[k] = peth_reliability(xs, events, pre, post)
        Ms = spatial_maps(xs, s.location, half, locs)
        null_space[k] = _rowcorr(Ms[0], Ms[1])
    thr_peth = np.nanpercentile(null_peth, 95, axis=0)
    thr_space = np.nanpercentile(null_space, 95, axis=0)

    full_peth = [peth(x, ev, pre, post) for ev in events]
    return {"x": x, "pre": pre, "post": post, "locs": locs, "peths": full_peth, "maps": M,
            "rel_peth": rel_peth, "thr_peth": thr_peth, "null_peth": null_peth,
            "rel_space": rel_space, "thr_space": thr_space, "null_space": null_space,
            "cells": np.flatnonzero(s.present)}


def _cv_heatmap(ax, odd, even, extent, vlim=1.0):
    """Cells ordered by the argmax of odd, showing even (both (n_bins, n_cells))."""
    order = np.argsort(np.argmax(odd, axis=0))
    im = ax.imshow(even[:, order].T, aspect="auto", cmap=colormap("diverging"), vmin=-vlim, vmax=vlim,
                   extent=extent, interpolation="nearest")
    return im


def plot_session(s, r, out_path):
    fig, axes = new_figure(2, 4, figsize=(22, 11))
    a = axes.ravel()
    for ax in a:
        style_axis(ax)
    ext = [-PRE_S, POST_S, r["x"].shape[1], 0]
    im = None
    for j, (name, P) in enumerate(zip(GOAL_NAMES, r["peths"])):
        ax = a[j]
        if len(P) >= 4:
            im = _cv_heatmap(ax, P[0::2].mean(0), P[1::2].mean(0), ext, vlim=0.8)
        else:
            ax.text(0.5, 0.5, f"{len(P)} events: too few for odd/even trials", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=INK_MUTED)
        ax.axvline(0, color=INK_PRIMARY, linewidth=1)
        title(ax, f"{name}_on (node {s.task[j]}): even trials, sorted by odd ({len(P)} events)")
        labels(ax, "time from poke (s)", "cell" if j == 0 else None)
    if im is not None:
        fig.colorbar(im, ax=a[2], label="mean z-scored activity")

    ax = a[3]
    M = r["maps"]
    if len(r["locs"]):
        im = _cv_heatmap(ax, M[0], M[1], [-0.5, len(r["locs"]) - 0.5, r["x"].shape[1], 0], vlim=0.8)
        ax.set_xticks(range(len(r["locs"])))
        ax.set_xticklabels([maze.location_name(l) for l in r["locs"]], rotation=90, fontsize=7.5)
        fig.colorbar(im, ax=ax, label="mean z-scored activity")
    title(ax, "Spatial maps: even trials, sorted by odd")
    labels(ax, "location")

    for j, name in enumerate(GOAL_NAMES):
        ax = a[4 + j]
        rel, null = r["rel_peth"][j], r["null_peth"][:, j].ravel()
        _rel_hist(ax, rel, null, r["thr_peth"][j], f"{name}_on PETH reliability")
    _rel_hist(a[7], r["rel_space"], r["null_space"].ravel(), r["thr_space"], "Spatial map reliability")
    save(fig, out_path, f"{s.label}  |  {r['x'].shape[1]} cells, task {'-'.join(map(str, s.task))}")


def _rel_hist(ax, rel, null, thr, ttl):
    bins = np.linspace(-1, 1, 41)
    ok = np.isfinite(rel)
    ax.hist(null[np.isfinite(null)], bins=bins, density=True, color=INK_MUTED, alpha=0.5, label="shift null")
    ax.hist(rel[ok], bins=bins, density=True, histtype="step", color=SERIES_1, linewidth=2, label="cells")
    frac = np.mean(rel[ok] > thr[ok]) if ok.any() else np.nan
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    title(ax, f"{ttl}: {frac:.0%} > own null p95")
    labels(ax, "odd-even correlation", "density")


def plot_summary(table, out_path, root):
    names = [l.split("/")[-1] for l in table.session]
    x = np.arange(len(names))
    fig, ax = new_figure(1, 1, figsize=(10, 5.5))
    style_axis(ax)
    series = [(f"frac_reliable_{n}", f"{n}_on PETH", c) for n, c in zip(GOAL_NAMES, GOAL_COLORS)]
    series.append(("frac_reliable_space", "spatial map", INK_PRIMARY))
    for i, (col, name, c) in enumerate(series):
        ax.plot(x + 0.1 * (i - 1.5), table[col], "o-", color=c, markersize=7, linewidth=1.2, label=name,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax.axhline(0.05, color=INK_MUTED, linestyle="--", linewidth=1, label="chance (5 %)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8)
    title(ax, "Cells with reproducible task / spatial tuning")
    labels(ax, y="fraction of cells")
    save(fig, out_path, f"Neural task checks  |  {root}")


def main():
    p = parser(__doc__, "neural_task_qc")
    p.add_argument("--n-shuffles", type=int, default=200)
    p.add_argument("--lag", type=int, default=0,
                   help="pair behaviour at t with activity at t - lag frames (exploration; default 0)")
    a = p.parse_args()
    root, out, sessions = setup(a, "neural_task_qc")
    (out / "sessions").mkdir(exist_ok=True)
    rows, cells = [], []
    for s in sessions:
        r = analyse(s, a.n_shuffles, a.lag)
        plot_session(s, r, out / "sessions" / f"{safe(s.label)}.png")
        row = {"session": s.label, "n_cells": r["x"].shape[1], "n_locs": len(r["locs"])}
        for j, n in enumerate(GOAL_NAMES):
            ok = np.isfinite(r["rel_peth"][j])
            row[f"frac_reliable_{n}"] = float(np.mean(r["rel_peth"][j][ok] > r["thr_peth"][j][ok]))
            row[f"median_rel_{n}"] = float(np.nanmedian(r["rel_peth"][j]))
        row["frac_reliable_space"] = float(np.mean(r["rel_space"] > r["thr_space"]))
        row["median_rel_space"] = float(np.nanmedian(r["rel_space"]))
        rows.append(row)
        cells.append(pd.DataFrame({"session": s.label, "roicat_col": r["cells"],
                                   **{f"rel_{n}": r["rel_peth"][j] for j, n in enumerate(GOAL_NAMES)},
                                   "rel_space": r["rel_space"], "null95_space": r["thr_space"]}))
        print(f"{s.label}: reliable PETH A/B/C "
              f"{row['frac_reliable_A']:.0%}/{row['frac_reliable_B']:.0%}/{row['frac_reliable_C']:.0%}, "
              f"spatial {row['frac_reliable_space']:.0%} (chance 5 %)")
    table = pd.DataFrame(rows)
    table.to_csv(out / "neural_task.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(out / "reliability.csv", index=False)
    plot_summary(table, out / "summary.png", root + (f"  |  activity lag {a.lag} frames" if a.lag else ""))
    print(f"Saved figures and tables to {out}")


if __name__ == "__main__":
    main()
