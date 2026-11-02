"""
Single-cell goal-progress tuning in raw ABC sessions processed by us
(suite2p on the raw tif; abc_data/raw.py), optionally after regressing a global
signal out of every cell.

    uv run python quality_control/abc_raw_tuning_qc.py /Volumes/behrens/Sandra/mini2p/Arya/RAW_GOOD
    uv run python quality_control/abc_raw_tuning_qc.py <root> --regress none   # or pop

Defaults: 10 bins, lag 0, background regressed out, no smoothing of the tuning
curves (heatmaps and all statistics); polar plots (display only) at 20 bins per
segment smoothed with sigma 2 bins.

Activity: F - 0.7 * Fneu of all suite2p ROIs (suite2p's iscell is not used), z-scored per cell.

Global-signal regression (--regress): each cell's z-scored trace minus its
least-squares fit to a global regressor:
    none         no regression (default)
    background   mean Fneu over all ROIs: the neuropil / whole-frame
                 fluorescence, which carries most of the shared signal in these
                 data (non-cellular brightness changes)
    pop          mean activity over cells: removes all shared signal, including
                 genuinely shared task-related activity, so interpret with care
pop_corr (correlation of each cell with the population mean) is reported before
and after.

Tuning: goal progress = time from one reward (A/B/C_on) to the next in --bins
equal bins (pooled over the three segments A->B, B->C, C->A). Tuning curve =
mean activity per bin; label at t paired with activity at t + --lag frames
(default 0; ~+8 frames compensates the ~1 s calcium delay seen in the lag scan).
Tuning strength = SD of the curve across bins. Null: the same after circularly
shifting all activity by a random >= 30 s (--n-shuffles); p = fraction of
shuffles >= observed. Preferred bin = argmax.

Heatmaps per session (tuning/<session>_<regress>.png), each row a cell's
curve normalised to its own min-max, rows sorted by preferred bin:
    significant cells, all trials
    cross-validated: sorted by the odd trials' preferred bin, showing the even
        trials' curves (a diagonal that survives is not a sorting artefact)
    segment-resolved: the same cells, curves computed separately in
        A->B | B->C | C->A, sorted by each cell's preferred bin in A->B; a
        diagonal that survives in B->C and C->A means the preferred progress
        generalises across segments (the A->B block is diagonal by construction)
    histogram of preferred bins (significant cells), tuning strength vs null
--smooth S: circular Gaussian smoothing of every curve used for heatmaps and
statistics (observed and null alike), sigma S bins; per-segment curves are
smoothed around the joined trial. --polar-bins / --polar-smooth only change the
polar plots (display: curves recomputed at --polar-bins per segment and smoothed
with sigma --polar-smooth of those bins; default 20 bins, sigma 2 = the same
time scale as sigma 1 at 10 bins); cells shown and statistics stay at --bins.

Segment generalisation (tuning/<...>_segments.png, cells.csv, summary.csv):
    segment consistency   mean correlation between a cell's A->B, B->C and C->A
                          curves; p vs the same circular-shift null
    preferred-phase match mean pairwise circular distance between the three
                          per-segment preferred phases (angle of the first circular
                          Fourier component of each curve, continuous, in bins);
                          p = fraction of shuffles with a distance <= observed
Both are reported for all cells (unbiased: 5 % expected by chance) and for the
goal-progress tuned cells (selected on the pooled curve, so expect some bias).
Compare against the shift null, not against fixed values: activity is
autocorrelated over seconds and a segment lasts ~5 s, so per-bin means drift
smoothly across the joined trial; in the null this slow, ~one-cycle-per-trial
structure puts the three per-segment phases ~120 deg apart (spread piles up at
the maximum, n_bins / 3) and makes segment curves slightly anti-correlated.

Frames per bin: a segment lasts a median 36-89 frames (4.8-11.8 s) in the AB_08
raw sessions, so with 10 bins one segment contributes ~4-9 frames per bin and
with 20 bins ~2-4.5 (< 1 in the shortest); the ~1 s calcium smoothing leaves
only ~5-10 independent steps per segment. 10 bins is about the useful limit.
Polar plots (tuning/<session>_<tag>_polar<page>.png): per significant cell, the
three segment curves joined around one trial (A at the top, B at 120 deg, C at
240 deg, clockwise), min-max normalised; goal-progress coding that generalises
across segments shows as three similar lobes at the same position in each third.
Near-direct segments (--max-excess-steps N, --max-wrong-pokes N; as in
abc_decoding_qc): goal progress only in reward-to-reward segments whose route
is at most N node steps longer than the shortest path / with at most N pokes
at non-goal ports; other frames are left out of every curve and statistic.
Routes need location per suite2p frame: preprocessing/track_position output in
--tracking (default ~/Desktop/maze_mini_qc/<name of root>/tracking) or, for
Arya's processed sessions without tracking, her maze_location moved to suite2p
frames by aligned_offset (behaviour labels, not affected by the 500-frame
shift of the neural data). File names and tables get _ex<N> / _pk<N>.

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of root>/tuning_qc):
tuning/*.png, cells.csv (per cell), summary.csv.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from abc_common import (PHASE_NAMES, direct_only, labels, new_figure, parser, safe, save, segment_route_stats,
                        setup, title)
from suite2p_io import default_out_dir
from suite2p_io import GRIDLINE, INK_MUTED, SERIES_1, SERIES_2, colormap, style_axis


def regress_out(x, g):
    """x (T, n) z-scored minus each column's least-squares fit to g (T,)."""
    g = (g - g.mean()) / g.std()
    beta = x.T @ g / (g @ g)
    return x - np.outer(g, beta), beta


def pop_corr(x):
    m = x.mean(axis=1)
    m = (m - m.mean()) / m.std()
    z = (x - x.mean(0)) / x.std(0)
    return z.T @ m / len(m)


def tuning(x, gp, n_bins, mask=None):
    """(n_bins, n_cells) mean activity per goal-progress bin."""
    T = np.full((n_bins, x.shape[1]), np.nan)
    for b in range(n_bins):
        m = gp == b
        if mask is not None:
            m &= mask
        if m.any():
            T[b] = x[m].mean(axis=0)
    return T


def apply_lag(x, lag):
    """Activity paired with labels at t is x[t + lag] (lag > 0: later activity)."""
    if lag == 0:
        return x
    xs = np.roll(x, -lag, axis=0)
    if lag > 0:
        xs[-lag:] = 0
    else:
        xs[:-lag] = 0
    return xs


def smooth_circular(T, sigma):
    """Gaussian smoothing (sigma in bins) along axis 0, wrapping around: goal
    progress is circular (the last bin of a segment is followed by the first bin
    of the next)."""
    if not sigma:
        return T
    n = T.shape[0]
    d = np.minimum(np.arange(n), n - np.arange(n))
    k = np.exp(-0.5 * (d / sigma) ** 2)
    k /= k.sum()
    K = np.stack([np.roll(k, i) for i in range(n)])     # row i: kernel centred on bin i
    return K @ np.nan_to_num(T)


def segment_curves(x, gp, seg, n_bins, sigma):
    """(3, n_bins, n_cells) per-segment curves; smoothed around the whole trial
    (A->B, B->C, C->A joined), so smoothing crosses rewards as time does."""
    S = np.concatenate([tuning(x, gp, n_bins, seg == j) for j in range(3)], axis=0)
    return smooth_circular(S, sigma).reshape(3, n_bins, -1)


def _colcorr(u, w):
    u = u - np.nanmean(u, 0)
    w = w - np.nanmean(w, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(u * w, 0) / np.sqrt(np.nansum(u ** 2, 0) * np.nansum(w ** 2, 0))


PAIRS = ((0, 1), (0, 2), (1, 2))


def preferred_phase(T, n_bins):
    """Continuous preferred position (in bins, 0..n_bins) of each curve along axis
    -2: the angle of its first circular Fourier component. Unlike argmax it uses the
    whole curve and has no ties, so shuffle p-values are not distorted."""
    ang = 2 * np.pi * (np.arange(n_bins) + 0.5) / n_bins
    shape = [1] * T.ndim
    shape[-2] = n_bins
    z = np.nansum((T - np.nanmean(T, axis=-2, keepdims=True)) * np.exp(1j * ang).reshape(shape), axis=-2)
    return (np.angle(z) % (2 * np.pi)) / (2 * np.pi) * n_bins


def segment_stats(Ts, n_bins):
    """Segment consistency (mean pairwise r between segment curves) and preferred-
    phase spread (mean pairwise circular distance, in bins, of the per-segment
    preferred phases from preferred_phase)."""
    cons = np.nanmean([_colcorr(Ts[i], Ts[j]) for i, j in PAIRS], axis=0)
    pref = preferred_phase(Ts, n_bins)                   # (3, n_cells)
    dist = []
    for i, j in PAIRS:
        d = np.abs(pref[i] - pref[j])
        dist.append(np.minimum(d, n_bins - d))
    return cons, np.mean(dist, axis=0), pref


def session_location(s, tracking_dir):
    """(n_frames,) location id per suite2p frame: track_position output in
    tracking_dir if present, else (Arya's processed sessions) maze_location of
    the aligned arrays placed at suite2p frame k + aligned_offset; NaN where
    unknown."""
    loc = np.full(s.n_frames, np.nan)
    path = Path(tracking_dir) / f"{safe(s.label)}_imaging_location.npy"
    if path.exists():
        v = np.load(path)[:s.n_frames]
        loc[:len(v)] = v
        return loc
    if hasattr(s, "session_files"):
        from abc_data import load_session
        ses = load_session(s.session_files)
        k = np.arange(ses.n_frames) + ses.aligned_offset
        ok = (k >= 0) & (k < s.n_frames)
        loc[k[ok]] = np.asarray(ses.location)[ok]
        return loc
    raise SystemExit(f"--max-excess-steps needs location per frame: {path} not found "
                     f"(run preprocessing/track_position.py or pass --tracking)")


def segment_mask(s, a, root):
    """(n_frames,) bool, frames in near-direct segments (--max-excess-steps /
    --max-wrong-pokes), or None if neither is set."""
    if a.max_excess_steps is None and a.max_wrong_pokes is None:
        return None
    if a.max_excess_steps is not None:
        tracking = os.path.expanduser(a.tracking or default_out_dir(root, "tracking"))
        loc = session_location(s, tracking)
    else:
        loc = np.full(s.n_frames, np.nan)
    gp = direct_only(np.zeros(s.n_frames, int), segment_route_stats(s, loc), a.max_excess_steps, a.max_wrong_pokes)
    return gp >= 0


def filter_note(a):
    """Figure-title text for the segment selection ('' if none)."""
    parts = ([f"<= {a.max_excess_steps} extra steps"] if a.max_excess_steps is not None else []) + \
        ([f"<= {a.max_wrong_pokes} wrong pokes"] if a.max_wrong_pokes is not None else [])
    return f"  |  segments {', '.join(parts)}" if parts else ""


def filter_tag(a):
    return (f"_ex{a.max_excess_steps}" if a.max_excess_steps is not None else "") + \
        (f"_pk{a.max_wrong_pokes}" if a.max_wrong_pokes is not None else "")


def analyse(s, n_bins, regress, lag, n_shuffles, smooth=0.0, seed=0, keep=None):
    """keep: (n_frames,) bool, frames whose goal-progress label is used (None = all)."""
    x = s.activity()
    x = (x - x.mean(0)) / x.std(0)
    pc_before = pop_corr(x)
    beta = None
    if regress != "none":
        x, beta = regress_out(x, s.global_signal(regress))
    pc_after = pop_corr(x)
    x = apply_lag(x, lag)
    gp = s.goal_progress(n_bins)
    if keep is not None:
        gp = np.where(keep, gp, -1)
    seg, _ = s.goal_segments()
    sm = lambda T: smooth_circular(T, smooth)
    T = sm(tuning(x, gp, n_bins))
    strength = np.nanstd(T, axis=0)
    T_seg = segment_curves(x, gp, seg, n_bins, smooth)
    cons, spread, pref_seg = segment_stats(T_seg, n_bins)
    rng = np.random.default_rng(seed)
    lo = int(30 * s.fs_measured)
    null = np.zeros((n_shuffles, x.shape[1]))
    null_cons = np.zeros_like(null)
    null_spread = np.zeros_like(null)
    for k in range(n_shuffles):
        xs = np.roll(x, rng.integers(lo, len(x) - lo), axis=0)
        null[k] = np.nanstd(sm(tuning(xs, gp, n_bins)), axis=0)
        null_cons[k], null_spread[k], _ = segment_stats(segment_curves(xs, gp, seg, n_bins, smooth), n_bins)
    p = (np.sum(null >= strength, axis=0) + 1) / (n_shuffles + 1)
    p_cons = (np.sum(null_cons >= cons, axis=0) + 1) / (n_shuffles + 1)
    p_spread = (np.sum(null_spread <= spread, axis=0) + 1) / (n_shuffles + 1)
    trial = s.trial_index()
    odd, even = (trial >= 0) & (trial % 2 == 1), (trial >= 0) & (trial % 2 == 0)
    T_odd, T_even = sm(tuning(x, gp, n_bins, odd)), sm(tuning(x, gp, n_bins, even))
    return {"x": x, "seg": seg, "keep": keep, "T": T, "T_odd": T_odd, "T_even": T_even, "T_seg": list(T_seg),
            "strength": strength, "null": null,
            "p": p, "pref": np.nanargmax(T, axis=0), "pref_odd": np.nanargmax(T_odd, axis=0),
            "pref_even": np.nanargmax(T_even, axis=0), "r_oe": _colcorr(T_odd, T_even),
            "seg_consistency": cons, "null_cons": null_cons, "p_cons": p_cons,
            "pref_spread": spread, "null_spread": null_spread, "p_spread": p_spread, "pref_seg": pref_seg,
            "pc_before": pc_before, "pc_after": pc_after, "beta": beta, "n_cells": x.shape[1]}


def plot_segments(s, r, n_bins, alpha, title_note, out_path):
    """Segment generalisation: consistency and preferred-bin spread vs null, for all
    cells (unbiased) and for the tuned ones."""
    sig = r["p"] < alpha
    fig, axes = new_figure(1, 3, figsize=(19, 5.5))
    for ax in axes:
        style_axis(ax)
    for ax, key, nkey, pkey, xl, ttl in (
            (axes[0], "seg_consistency", "null_cons", "p_cons", "mean r between segment curves",
             "Segment consistency"),
            (axes[1], "pref_spread", "null_spread", "p_spread", "mean |preferred-phase difference| (bins, circular)",
             "Preferred phase across segments")):
        allv = np.r_[r[nkey].ravel(), r[key]]
        bins = np.linspace(np.nanmin(allv), np.nanmax(allv), 30)
        ax.hist(r[nkey].ravel(), bins=bins, density=True, color=INK_MUTED, alpha=0.5, label="shift null")
        ax.hist(r[key], bins=bins, density=True, histtype="step", color=SERIES_1, linewidth=2, label="all cells")
        if sig.any():
            ax.hist(r[key][sig], bins=bins, density=True, histtype="step", color=SERIES_2, linewidth=2,
                    label="goal-progress tuned cells")
        f_all, f_sig = np.mean(r[pkey] < alpha), (np.mean(r[pkey][sig] < alpha) if sig.any() else np.nan)
        ax.legend(frameon=False, fontsize=8)
        title(ax, f"{ttl}: p<{alpha} in {f_all:.0%} of all cells, {f_sig:.0%} of tuned (chance {alpha:.0%})")
        labels(ax, xl, "density")
    ax = axes[2]
    ax.scatter(-np.log10(r["p"]), -np.log10(r["p_cons"]), s=14, color=SERIES_1, alpha=0.6, edgecolor="none")
    for v in (-np.log10(alpha),):
        ax.axvline(v, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.axhline(v, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    title(ax, "Pooled tuning vs segment consistency (per cell)")
    labels(ax, "-log10 p, pooled goal-progress tuning", "-log10 p, segment consistency")
    save(fig, out_path, f"{s.label}  |  does goal-progress tuning generalise across A->B, B->C, C->A?  |  {title_note}")


def _norm(T):
    lo, hi = np.nanmin(T, 0), np.nanmax(T, 0)
    return (T - lo) / np.where(hi > lo, hi - lo, 1)


def plot_session(s, r, n_bins, alpha, regress, lag, out_path):
    sig = np.flatnonzero(r["p"] < alpha)
    fig, axes = new_figure(2, 3, figsize=(19, 11))
    a = axes.ravel()
    for ax in a:
        style_axis(ax)
    cmap = colormap()
    centres = (np.arange(n_bins) + 0.5) / n_bins

    def heat(ax, T, order, ttl, extent=(0, 1)):
        im = ax.imshow(_norm(T)[:, order].T, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                       extent=[extent[0], extent[1], len(order), 0])
        title(ax, ttl)
        return im

    order = sig[np.argsort(r["pref"][sig], kind="stable")]
    if len(order):
        im = heat(a[0], r["T"], order, f"Significant cells (p < {alpha}), all trials, sorted by preferred bin")
        fig.colorbar(im, ax=a[0], label="normalised activity (min-max per cell)")
        order_cv = sig[np.argsort(r["pref_odd"][sig], kind="stable")]
        heat(a[1], r["T_even"], order_cv, "Cross-validated: even trials, sorted by odd trials")
        S = np.concatenate([_norm(T) for T in r["T_seg"]], axis=0)
        T_ab = r["T_seg"][0][:, sig]
        pref_ab = np.argmax(np.where(np.isnan(T_ab), -np.inf, T_ab), axis=0)
        order_ab = sig[np.argsort(pref_ab, kind="stable")]
        a[2].imshow(S[:, order_ab].T, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                    extent=[0, 3, len(order_ab), 0])
        for j in (1, 2):
            a[2].axvline(j, color="white", linewidth=2)
        a[2].set_xticks([0.5, 1.5, 2.5])
        a[2].set_xticklabels(PHASE_NAMES)
        title(a[2], f"Per segment (same cells), sorted by {PHASE_NAMES[0]} preferred bin, each normalised")
    for ax in a[:2]:
        labels(ax, "goal progress (fraction of reward-to-reward time)", "cell")

    ax = a[3]
    if len(sig):
        cnt = np.bincount(r["pref"][sig], minlength=n_bins)
        ax.bar(centres, cnt, width=0.9 / n_bins, color=SERIES_1, edgecolor="white")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    title(ax, f"Preferred bin of significant cells ({len(sig)}/{r['n_cells']}, chance {alpha:.0%})")
    labels(ax, "goal progress", "cells")

    ax = a[4]
    bins = np.linspace(0, max(np.nanmax(r["strength"]), np.nanmax(r["null"])) * 1.05, 40)
    ax.hist(r["null"].ravel(), bins=bins, density=True, color=INK_MUTED, alpha=0.5, label="shift null (all cells)")
    ax.hist(r["strength"], bins=bins, density=True, histtype="step", color=SERIES_1, linewidth=2, label="cells")
    ax.legend(frameon=False, fontsize=8)
    title(ax, "Tuning strength (SD of curve across bins)")
    labels(ax, "tuning strength (z units)", "density")

    ax = a[5]
    d = np.abs(r["pref_odd"] - r["pref_even"])
    d = np.minimum(d, n_bins - d)
    for keep, c, name in ((r["p"] < alpha, SERIES_1, "significant"), (r["p"] >= alpha, SERIES_2, "not significant")):
        if keep.any():
            ax.hist(d[keep], bins=np.arange(n_bins // 2 + 2) - 0.5, density=True, histtype="step", linewidth=2,
                    color=c, label=f"{name} (median r odd/even {np.nanmedian(r['r_oe'][keep]):.2f})")
    ax.legend(frameon=False, fontsize=8)
    title(ax, "Preferred bin, odd vs even trials")
    labels(ax, "|difference| (bins, circular)", "fraction of cells")
    note = {"none": "no regression", "background": "background (mean Fneu) regressed out",
            "pop": "population mean regressed out"}[regress]
    save(fig, out_path, f"{s.label}  |  goal-progress tuning, {n_bins} bins  |  {note}  |  activity lag {lag} frames  |  "
                        f"pop_corr {np.median(r['pc_before']):.2f} -> {np.median(r['pc_after']):.2f}"
                        f"{r.get('filter_note', '')}")


def plot_polar(s, r, n_bins, alpha, regress, lag, out_stem, sigma=2.0, polar_bins=20, per_page=30, ncols=6):
    """One polar panel per significant cell: the three segment curves (A->B, B->C,
    C->A) joined around a circle (A at the top, B at 120 deg, C at 240 deg, one trial
    = one turn), min-max normalised per cell. Cells ordered by preferred bin.
    Returns the written file paths."""
    import matplotlib.pyplot as plt
    sig = np.flatnonzero(r["p"] < alpha)
    order = sig[np.argsort(r["pref"][sig], kind="stable")]
    # display smoothing only (sigma bins, around the joined trial); statistics use r as computed
    # display only: per-segment curves recomputed at polar_bins and smoothed (sigma in polar
    # bins, around the joined trial); which cells are shown and all statistics come from r
    gp_polar = s.goal_progress(polar_bins)
    if r.get("keep") is not None:
        gp_polar = np.where(r["keep"], gp_polar, -1)
    T_seg = segment_curves(r["x"], gp_polar, r["seg"], polar_bins, 0)
    S = smooth_circular(T_seg.reshape(3 * polar_bins, -1), sigma)    # (3 * polar_bins, n_cells)
    n_stat_bins, n_bins = n_bins, polar_bins
    theta = (np.arange(3 * n_bins) + 0.5) / (3 * n_bins) * 2 * np.pi
    note = {"none": "no regression", "background": "background regressed out", "pop": "population mean regressed out"}[regress]
    paths = []
    for page, start in enumerate(range(0, max(len(order), 1), per_page)):
        cells = order[start:start + per_page]
        nrows = max(1, int(np.ceil(len(cells) / ncols)))
        fig = plt.figure(figsize=(3.1 * ncols, 3.3 * nrows + 0.6))
        fig.patch.set_facecolor("#fcfcfb")
        for i, c in enumerate(cells):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="polar")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            v = S[:, c]
            v = (v - np.nanmin(v)) / max(np.nanmax(v) - np.nanmin(v), 1e-9)
            ax.plot(np.r_[theta, theta[:1]], np.r_[v, v[:1]], color=SERIES_1, linewidth=1.4)
            ax.fill(np.r_[theta, theta[:1]], np.r_[v, v[:1]], color=SERIES_1, alpha=0.15)
            for k, g in enumerate("ABC"):
                ang = k * 2 * np.pi / 3
                ax.plot([ang, ang], [0, 1.05], color=INK_MUTED, linewidth=0.8)
            ax.set_xticks([0, 2 * np.pi / 3, 4 * np.pi / 3])
            ax.set_xticklabels(["A", "B", "C"], fontsize=8)
            ax.set_yticks([])
            ax.set_ylim(0, 1.05)
            ax.set_title(f"cell {np.flatnonzero(s.cells)[c]}  pref {r['pref'][c] + 1}/{n_stat_bins}  p={r['p'][c]:.3f}",
                         fontsize=7.5, color="#52514e")
        fig.suptitle(f"{s.label}  |  tuned cells ({len(order)}), curves per segment around one trial "
                     f"(A top, B 120°, C 240°, clockwise)  |  display: {n_bins} bins/segment, sigma {sigma:g} bins  |  "
                     f"cells and stats: {n_stat_bins} bins  |  {note}  |  lag {lag}{r.get('filter_note', '')}"
                     + (f"  |  page {page + 1}" if len(order) > per_page else ""), fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = f"{out_stem}_polar{page + 1}.png"
        fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(path)
    return paths


def add_tuning_args(p):
    """Tuning options shared with abc_processed_tuning_qc (everything except the lag)."""
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--regress", choices=["none", "background", "pop"], default="background")
    p.add_argument("--n-shuffles", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--smooth", type=float, default=0.0,
                   help="circular Gaussian smoothing of tuning curves (heatmaps and statistics), sigma in bins "
                        "(default 0 = none)")
    p.add_argument("--polar-bins", type=int, default=20,
                   help="bins per segment for the polar plots only (display; default 20)")
    p.add_argument("--polar-smooth", type=float, default=2.0,
                   help="display-only smoothing of the polar plots, sigma in polar bins (default 2; 0 = none)")
    p.add_argument("--max-excess-steps", type=int,
                   help="only segments at most N node steps longer than the shortest path")
    p.add_argument("--max-wrong-pokes", type=int, help="only segments with at most N pokes at non-goal ports")
    p.add_argument("--tracking", help="track_position output folder (default ~/Desktop/maze_mini_qc/<name of "
                                      "root>/tracking; processed sessions fall back to Arya's maze_location)")
    return p


def run_session(s, a, lag, out, keep=None):
    """Analyse one session at one lag, write its three figures; returns (summary row,
    per-cell DataFrame, analyse() result). keep: segment_mask()."""
    r = analyse(s, a.bins, a.regress, lag, a.n_shuffles, a.smooth, keep=keep)
    r["filter_note"] = filter_note(a)
    sig = r["p"] < a.alpha
    tag = f"{a.regress}_lag{lag}_bins{a.bins}_smooth{a.smooth:g}{filter_tag(a)}"
    plot_session(s, r, a.bins, a.alpha, a.regress, lag, out / "tuning" / f"{safe(s.label)}_{tag}.png")
    plot_polar(s, r, a.bins, a.alpha, a.regress, lag,
               str(out / "tuning" / f"{safe(s.label)}_{tag}_polar{a.polar_bins}bins_sigma{a.polar_smooth:g}"),
               a.polar_smooth, a.polar_bins)
    plot_segments(s, r, a.bins, a.alpha, f"{a.bins} bins, {a.regress} regression, smoothing sigma {a.smooth:g} bins, "
                  f"lag {lag}{filter_note(a)}", out / "tuning" / f"{safe(s.label)}_{tag}_segments.png")
    d = np.abs(r["pref_odd"] - r["pref_even"])
    d = np.minimum(d, a.bins - d)
    n_seg = len(np.unique(s.goal_segments()[1][(keep if keep is not None else np.ones(s.n_frames, bool))
                                              & (s.goal_progress(a.bins) >= 0)]))
    row = {"session": s.label, "regress": a.regress, "lag": lag, "bins": a.bins, "n_cells": r["n_cells"],
           "max_excess_steps": a.max_excess_steps, "max_wrong_pokes": a.max_wrong_pokes, "n_segments": n_seg,
           "n_significant": int(sig.sum()), "frac_significant": float(sig.mean()),
           "median_pop_corr_before": float(np.median(r["pc_before"])),
           "median_pop_corr_after": float(np.median(r["pc_after"])),
           "median_r_oddeven_sig": float(np.nanmedian(r["r_oe"][sig])) if sig.any() else np.nan,
           "frac_pref_within1_sig": float(np.mean(d[sig] <= 1)) if sig.any() else np.nan,
           "frac_pref_within1_nonsig": float(np.mean(d[~sig] <= 1)) if (~sig).any() else np.nan,
           "frac_seg_consistent_all": float(np.mean(r["p_cons"] < a.alpha)),
           "frac_seg_consistent_tuned": float(np.mean(r["p_cons"][sig] < a.alpha)) if sig.any() else np.nan,
           "frac_pref_match_all": float(np.mean(r["p_spread"] < a.alpha)),
           "frac_pref_match_tuned": float(np.mean(r["p_spread"][sig] < a.alpha)) if sig.any() else np.nan,
           "n_tuned_and_seg_consistent": int(np.sum(sig & (r["p_cons"] < a.alpha)))}
    cells = pd.DataFrame({"session": s.label, "regress": a.regress, "lag": lag,
                          "cell": np.flatnonzero(s.cells), "strength": r["strength"], "p": r["p"],
                          "pref_bin": r["pref"], "pref_odd": r["pref_odd"], "pref_even": r["pref_even"],
                          "r_oddeven": r["r_oe"], "seg_consistency": r["seg_consistency"],
                          "p_seg_consistency": r["p_cons"], "pref_spread_bins": r["pref_spread"],
                          "p_pref_spread": r["p_spread"], "pref_AB": r["pref_seg"][0],
                          "pref_BC": r["pref_seg"][1], "pref_CA": r["pref_seg"][2],
                          "pop_corr_before": r["pc_before"], "pop_corr_after": r["pc_after"]})
    print(f"{s.label} [{a.regress}, lag {lag}{filter_tag(a)}, {n_seg} segments]: {sig.sum()}/{r['n_cells']} significant ({sig.mean():.0%}, "
          f"chance {a.alpha:.0%}); pop_corr {row['median_pop_corr_before']:.2f} -> "
          f"{row['median_pop_corr_after']:.2f}; preferred bin odd/even within 1 bin: "
          f"{row['frac_pref_within1_sig']:.0%} of significant vs {row['frac_pref_within1_nonsig']:.0%} "
          f"of others (chance {3 / a.bins:.0%}) | segment-consistent (p<{a.alpha}): "
          f"{row['frac_seg_consistent_all']:.0%} of all, {row['frac_seg_consistent_tuned']:.0%} of tuned; "
          f"preferred phase matches across segments: {row['frac_pref_match_all']:.0%} of all, "
          f"{row['frac_pref_match_tuned']:.0%} of tuned (chance {a.alpha:.0%})", flush=True)
    return row, cells, r


def main():
    p = add_tuning_args(parser(__doc__, "tuning_qc"))
    p.add_argument("--lag", type=int, default=0, help="pair labels at t with activity at t + lag frames")
    a = p.parse_args()
    root, out, sessions = setup(a, "tuning_qc")
    (out / "tuning").mkdir(exist_ok=True)
    rows, cells = [], []
    for s in sessions:
        row, c, _ = run_session(s, a, a.lag, out, segment_mask(s, a, root))
        rows.append(row)
        cells.append(c)
    sfx = f"{a.regress}_lag{a.lag}_bins{a.bins}_smooth{a.smooth:g}{filter_tag(a)}"
    pd.DataFrame(rows).to_csv(out / f"summary_{sfx}.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(out / f"cells_{sfx}.csv", index=False)
    print(f"Saved figures and tables to {out}")


if __name__ == "__main__":
    main()
