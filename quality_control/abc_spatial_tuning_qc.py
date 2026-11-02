"""
Spatially tuned cells on the maze: within each session (odd vs even trials) and,
for Arya's processed sessions, across sessions (do ROICaT-tracked cells keep the
same spatial map from one session to the next, more than expected from shuffled
cell identities?). Maps of the cells that pass are drawn as maze heatmaps.

    uv run python quality_control/abc_spatial_tuning_qc.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding
    uv run python quality_control/abc_spatial_tuning_qc.py <root> --lags 0 -250 -500
    uv run python quality_control/abc_spatial_tuning_qc.py /Volumes/behrens/Sandra/mini2p/Arya/RAW_GOOD --lags 0 8

Processed layout: Arya's aligned ROICaT activity (columns = the same cell in
every session, NaN = absent) and maze_location. Raw layout (abc_data.raw, our
suite2p): F - 0.7 Fneu of all ROIs and the location per suite2p frame from
preprocessing/track_position.py (<session>_imaging_location.npy in --tracking,
default ~/Desktop/maze_mini_qc/<name of root>/tracking); raw sessions have no
cross-session cell matching, so only the within-session test runs. Activity is
z-scored per cell and session with the background (mean Fneu over ROIs)
regressed out (--regress none to skip), as recommended for single-cell
analyses here (CLAUDE.md). Locations: nodes 1-9 and edges (unknown frames left
out); locations with fewer than --min-frames frames are left out of a map.

Lag (as in abc_processed_tuning_qc): behaviour at frame t is paired with
activity at frame t + lag; frames without a partner are dropped. The
~500-frame shift in the processed sessions (CLAUDE.md, known findings) is
lag = -500: default --lags 0 -500 there, 0 for raw sessions.

Within session: maps from odd and even trials (each location needs
--min-frames in both), reliability = their correlation; null = each cell's
activity circularly shifted against behaviour by >= 30 s (--n-shuffles); a cell
is 'reliable' if p < --alpha.

Across sessions (processed only): map of a cell in a session = mean activity
per location; consistency = mean over session pairs (both containing the cell)
of the correlation of its two maps, over locations valid in both. Null =
shuffled identities: in each session pair the partner cell is drawn at random
from the cells present in both sessions, which keeps every map and any spatial
structure shared by all cells, and only breaks the identity.

Location and task are entangled: goal ports (marked on every map) are visited
with reward and the animal slows there, so goal- or speed-locked activity looks
spatial within a session. A/B/C goals move between sessions, so across sessions
it does not, unless a goal stays on the same node.

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of root>/spatial_qc):
    within/<session>_lag<L>_<page>.png  maze heatmaps of the reliable cells
                                (all trials | odd | even), up to --max-cells
    within_summary.png/.csv     fraction reliable per session and lag, r vs null
    within_cells.csv            per cell: r, p
    summary.png                 (processed) per lag: cross-session consistency
                                vs identity null, mean r per session pair
    cells_lag<L>_<page>.png     (processed) maze heatmaps of the consistent cells
                                (rows) in every session (columns)
    cells.csv, pairs.csv        (processed) per cell / session pair and lag
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import os
from pathlib import Path

from abc_common import GOAL_COLORS, GOAL_NAMES, is_raw_root, labels, new_figure, parser, safe, save, setup, title
from abc_data import maze
from abc_raw_tuning_qc import regress_out
from suite2p_io import default_out_dir, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SERIES_1, SERIES_2, colormap, style_axis

LOCS = tuple(l for l in maze.LOCATIONS if l not in maze.ABSENT_EDGES)


def session_activity(s, regress):
    """(T, n_roicat) z-scored activity, NaN columns for absent cells."""
    x = np.asarray(s.activity, dtype=np.float64)
    pres = s.present
    z = np.full_like(x, np.nan)
    xp = np.nan_to_num((x[:, pres] - np.nanmean(x[:, pres], 0)) / np.nanstd(x[:, pres], 0))
    if regress == "background":
        g = np.nanmean(s.suite2p_dff_aligned("Fneu"), axis=1)
        g = np.where(np.isfinite(g), g, np.nanmean(g))
        xp = regress_out(xp, g)[0]
        xp = (xp - xp.mean(0)) / xp.std(0)
    z[:, pres] = xp
    return z


def spatial_maps(s, x, lag, min_frames):
    """(n_locs, n_roicat) mean activity per location with activity at t + lag;
    rows of locations with < min_frames frames are NaN. Also the frame counts."""
    T = s.n_frames
    t = np.arange(T)
    valid = (t + lag >= 0) & (t + lag < T)
    loc = s.location[valid]
    xs = x[t[valid] + lag]
    M = np.full((len(LOCS), x.shape[1]), np.nan)
    n = np.zeros(len(LOCS), int)
    for j, l in enumerate(LOCS):
        m = loc == l
        n[j] = m.sum()
        if n[j] >= min_frames:
            M[j] = xs[m].mean(0)
    return M, n


def _pair_corr(Ma, Mb, cells):
    """(n, n) correlation of the maps of `cells` in a (rows) vs b (columns) over
    the locations valid in both."""
    ok = np.all(np.isfinite(Ma[:, cells]), 1) & np.all(np.isfinite(Mb[:, cells]), 1)
    if ok.sum() < 4:
        return None
    a, b = Ma[ok][:, cells], Mb[ok][:, cells]
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    return np.nan_to_num(a.T @ b / ok.sum())


def consistency(maps, present, n_shuffles, seed=0):
    """Per cell: mean map correlation across session pairs, identity-shuffle null
    (n_shuffles, n_cells), number of session pairs; per pair: a dict row."""
    n_cells = maps[0].shape[1]
    rng = np.random.default_rng(seed)
    tot = np.zeros(n_cells)
    null = np.zeros((n_shuffles, n_cells))
    cnt = np.zeros(n_cells)
    pairs = []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            cells = np.flatnonzero(present[i] & present[j])
            if len(cells) < 3:
                continue
            C = _pair_corr(maps[i], maps[j], cells)
            if C is None:
                continue
            d = np.diag(C)
            tot[cells] += d
            cnt[cells] += 1
            nm = []
            for k in range(n_shuffles):
                # partner in session j drawn from the other shared cells
                perm = rng.permutation(len(cells))
                same = perm == np.arange(len(cells))
                perm[same] = (perm[same] + 1) % len(cells)
                v = C[np.arange(len(cells)), perm]
                null[k, cells] += v
                nm.append(v.mean())
            pairs.append({"a": i, "b": j, "shared": len(cells), "mean_r": d.mean(),
                          "null_mean": float(np.mean(nm)), "null_p95": float(np.percentile(nm, 95))})
    with np.errstate(invalid="ignore", divide="ignore"):
        cons = tot / cnt
        null = null / cnt
    return cons, null, cnt, pairs


# ---- maze heatmap ----
def _node_xy(n):
    return (n - 1) % 3, (n - 1) // 3


NODE, BRIDGE = 0.5, 0.18


def draw_maze(ax, values, vmax, cmap, task=None, counts=None):
    """Maze schematic (node 1 bottom left, as in the video) with each location
    coloured by its value; NaN locations hatched grey."""
    ax.set_aspect("equal")
    ax.axis("off")
    for j, l in enumerate(LOCS):
        if l in maze.EDGES:
            (xa, ya), (xb, yb) = (_node_xy(n) for n in maze.EDGES[l])
            if ya == yb:
                r = (min(xa, xb) + NODE / 2, ya - BRIDGE / 2, 1 - NODE, BRIDGE)
            else:
                r = (xa - BRIDGE / 2, min(ya, yb) + NODE / 2, BRIDGE, 1 - NODE)
        else:
            x0, y0 = _node_xy(l)
            r = (x0 - NODE / 2, y0 - NODE / 2, NODE, NODE)
        v = values[j]
        if np.isfinite(v):
            ax.add_patch(Rectangle(r[:2], r[2], r[3], facecolor=cmap(0.5 + 0.5 * np.clip(v / vmax, -1, 1)),
                                   edgecolor="white", linewidth=0.6))
        else:
            ax.add_patch(Rectangle(r[:2], r[2], r[3], facecolor="none", edgecolor=GRIDLINE, hatch="////",
                                   linewidth=0.6))
    if task is not None:
        for name, node, c in zip(GOAL_NAMES, task, GOAL_COLORS):
            x0, y0 = _node_xy(node)
            ax.add_patch(Rectangle((x0 - NODE / 2, y0 - NODE / 2), NODE, NODE, facecolor="none", edgecolor=c,
                                   linewidth=2))
            ax.text(x0 - NODE / 2 + 0.04, y0 + NODE / 2 - 0.04, name, color=c, fontsize=7, fontweight="bold",
                    ha="left", va="top")
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.3, 2.3)


def plot_cells(sessions, maps, present, res, lag, alpha, max_cells, out, per_page=8):
    sig = np.flatnonzero(res["p"] < alpha)
    sig = sig[np.argsort(-res["cons"][sig])][:max_cells]
    cmap = colormap("diverging")
    n_s = len(sessions)
    pages = [sig[i:i + per_page] for i in range(0, len(sig), per_page)]
    for pg, cells in enumerate(pages):
        fig, axes = new_figure(len(cells), n_s + 1, figsize=(1.9 * n_s + 1.6, 2.0 * len(cells) + 0.9),
                               squeeze=False, gridspec_kw={"width_ratios": [1] * n_s + [0.12]})
        for r, c in enumerate(cells):
            vals = [maps[k][:, c] for k in range(n_s) if present[k][c]]
            vmax = max(np.nanmax(np.abs(v)) for v in vals) or 1
            for k, s in enumerate(sessions):
                ax = axes[r, k]
                if present[k][c]:
                    draw_maze(ax, maps[k][:, c], vmax, cmap, s.task)
                else:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "absent", transform=ax.transAxes, ha="center", va="center",
                            color=INK_MUTED, fontsize=8)
                if r == 0:
                    title(ax, s.label.split("/")[-1])
            axes[r, 0].text(-0.08, 0.5, f"cell {c}\nr = {res['cons'][c]:.2f}\np = {res['p'][c]:.3f}\n"
                            f"{sum(pr[c] for pr in present)} sessions", transform=axes[r, 0].transAxes, ha="right", va="center",
                            fontsize=8, color=INK_SECONDARY)
            cax = axes[r, -1]
            grad = np.linspace(-1, 1, 64)[:, None]
            cax.imshow(grad, aspect="auto", cmap=cmap, origin="lower", extent=[0, 1, -vmax, vmax])
            cax.set_xticks([])
            cax.yaxis.tick_right()
            cax.tick_params(labelsize=7, colors=INK_SECONDARY)
            cax.set_ylabel("z", fontsize=7, color=INK_SECONDARY)
            cax.yaxis.set_label_position("right")
        fig.subplots_adjust(left=0.1, right=0.95, top=1 - 0.9 / (2.0 * len(cells) + 0.9), bottom=0.02,
                            wspace=0.08, hspace=0.25)
        fig.suptitle(f"Cross-session consistent spatial maps, lag {lag} frames (activity − behaviour)\n"
                     f"mean activity per location (z), hatched = < min frames; boxes = A/B/C goals  |  "
                     f"page {pg + 1}/{len(pages)}", fontsize=10, x=0.01, ha="left", color=INK_PRIMARY)
        fig.savefig(out / f"cells_lag{lag}_{pg + 1}.png", dpi=130, facecolor=fig.get_facecolor())
        import matplotlib.pyplot as plt
        plt.close(fig)
    return len(sig)


def plot_summary(sessions, results, lags, alpha, out_path, root):
    n_l = len(lags)
    fig, axes = new_figure(2, max(n_l, 2), figsize=(5.5 * max(n_l, 2), 9.5), squeeze=False)
    for ax in axes.ravel():
        style_axis(ax)
    for i, lag in enumerate(lags):
        r = results[lag]
        ok = r["n"] >= 1
        ax = axes[0, i]
        bins = np.linspace(-1, 1, 41)
        ax.hist(r["null"][:, ok].ravel(), bins=bins, density=True, color=INK_MUTED, alpha=0.5,
                label="identity shuffle")
        ax.hist(r["cons"][ok], bins=bins, density=True, histtype="step", color=SERIES_1, linewidth=2,
                label="same cell")
        frac = np.mean(r["p"][ok] < alpha)
        title(ax, f"lag {lag}: {ok.sum()} cells in ≥ 2 sessions\n{frac:.1%} consistent (p < {alpha}), "
                  f"mean r {np.nanmean(r['cons'][ok]):.3f} vs null {np.nanmean(r['null'][:, ok]):.3f}")
        labels(ax, "cross-session map correlation (mean over session pairs)", "density")
        ax.legend(frameon=False, fontsize=8)
    # pair matrix for each lag, true mean r - null mean
    n_s = len(sessions)
    short = [s.label.split("/")[-1][5:] for s in sessions]
    for i, lag in enumerate(lags):
        ax = axes[1, i]
        D = np.full((n_s, n_s), np.nan)
        for p in results[lag]["pairs"]:
            D[p["a"], p["b"]] = D[p["b"], p["a"]] = p["mean_r"] - p["null_mean"]
        v = np.nanmax(np.abs(D)) if np.isfinite(D).any() else 1
        im = ax.imshow(D, cmap=colormap("diverging"), vmin=-v, vmax=v)
        for p in results[lag]["pairs"]:
            star = "*" if p["mean_r"] > p["null_p95"] else ""
            for a, b in ((p["a"], p["b"]), (p["b"], p["a"])):
                ax.text(b, a, f"{D[a, b]:.2f}{star}", ha="center", va="center", fontsize=7, color=INK_PRIMARY)
        ax.set_xticks(range(n_s))
        ax.set_xticklabels(short, rotation=45, fontsize=8)
        ax.set_yticks(range(n_s))
        ax.set_yticklabels(short, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8, label="mean r of shared cells − null mean")
        title(ax, f"lag {lag}: session pairs (* > null 95th pct.)")
    for ax in axes[0, n_l:]:
        ax.axis("off")
    for ax in axes[1, n_l:]:
        ax.axis("off")
    save(fig, out_path, f"Cross-session spatial-map consistency vs shuffled identities  |  {root}")


# ---- within session (both layouts; the only test for raw sessions) ----
def raw_inputs(s, regress, tracking_dir):
    """Raw session: (T, n_cells) z-scored activity (F - 0.7 Fneu of all ROIs, global
    signal regressed), location per suite2p frame from track_position, trial index."""
    path = Path(tracking_dir) / f"{safe(s.label)}_imaging_location.npy"
    if not path.exists():
        raise SystemExit(f"{path} not found: run preprocessing/track_position.py on this root first "
                         f"(or pass --tracking)")
    loc = np.load(path)
    x = s.activity()
    x = (x - x.mean(0)) / x.std(0)
    if regress == "background":
        x = regress_out(x, s.global_signal("background"))[0]
        x = (x - x.mean(0)) / x.std(0)
    return x, loc[:len(x)], s.trial_index()


def processed_inputs(s, x):
    """Processed session: present ROICaT columns of session_activity, Arya's
    maze_location, trial index from trial_times."""
    tr = np.full(s.n_frames, -1)
    for i, (a, *_, end) in enumerate(s.trial_times):
        tr[a:end] = i
    return x[:, s.present], s.location, tr


def _onehot(loc, mask):
    """(T, n_locs) location indicator restricted to mask, and counts."""
    L = np.stack([(loc == l) & mask for l in LOCS], axis=1).astype(np.float64)
    return L, L.sum(0)


def within_session(x, loc, trial, lag, min_frames, n_shuffles, min_shift, seed=0):
    """Odd/even-trial map reliability per cell vs a per-cell circular-shift null.
    Behaviour at t is paired with activity at t + lag. Returns maps (all trials),
    odd/even maps, reliability, null (n_shuffles, n_cells), p."""
    T, n = x.shape
    t = np.arange(T)
    valid = (t + lag >= 0) & (t + lag < T) & (trial >= 0) & (loc != maze.UNKNOWN)
    src = np.clip(t + lag, 0, T - 1)
    halves = [valid & (trial % 2 == h) for h in (0, 1)]
    Ls = [_onehot(loc, m) for m in [valid] + halves]
    ok = (Ls[1][1] >= min_frames) & (Ls[2][1] >= min_frames)

    def maps(xs):
        return [np.where((c >= min_frames)[:, None], (L.T @ xs) / np.maximum(c, 1)[:, None], np.nan) for L, c in Ls]

    def rel(M1, M2):
        a, b = M1[ok], M2[ok]
        a = (a - a.mean(0)) / a.std(0)
        b = (b - b.mean(0)) / b.std(0)
        return (a * b).mean(0)

    xl = x[src]
    M, Mo, Me = maps(xl)
    r = rel(Mo, Me)
    rng = np.random.default_rng(seed)
    null = np.zeros((n_shuffles, n))
    cols = np.arange(n)
    for k in range(n_shuffles):
        sh = rng.integers(min_shift, T - min_shift, n)
        xs = x[(src[:, None] - sh[None, :]) % T, cols[None, :]]
        _, a_, b_ = maps(xs)
        null[k] = rel(a_, b_)
    p = (1 + np.sum(null >= r, axis=0)) / (1 + n_shuffles)
    return {"M": M, "Mo": Mo, "Me": Me, "r": r, "null": null, "p": p, "n_locs": int(ok.sum())}


def plot_within_cells(s, res, cell_ids, lag, alpha, max_cells, out_stem, per_page=10, ncols=2):
    """Maze heatmaps of the reliable cells: all trials | odd | even, two cells per row."""
    sig = np.flatnonzero(res["p"] < alpha)
    sig = sig[np.argsort(-res["r"][sig])][:max_cells]
    cmap = colormap("diverging")
    pages = [sig[i:i + per_page] for i in range(0, len(sig), per_page)]
    for pg, cells in enumerate(pages):
        nrows = int(np.ceil(len(cells) / ncols))
        fig, axes = new_figure(nrows, 4 * ncols, figsize=(2.0 * 4 * ncols, 2.1 * nrows + 0.9), squeeze=False,
                               gridspec_kw={"width_ratios": [1, 1, 1, 0.1] * ncols})
        for ax in axes.ravel():
            ax.axis("off")
        for k, c in enumerate(cells):
            row, col = divmod(k, ncols)
            vmax = np.nanmax(np.abs(np.r_[res["M"][:, c], res["Mo"][:, c], res["Me"][:, c]])) or 1
            for j, (key, name) in enumerate((("M", "all trials"), ("Mo", "odd"), ("Me", "even"))):
                ax = axes[row, 4 * col + j]
                draw_maze(ax, res[key][:, c], vmax, cmap, s.task)
                title(ax, f"cell {cell_ids[c]}: r = {res['r'][c]:.2f}, p = {res['p'][c]:.3f}" if j == 0 else name)
            cax = axes[row, 4 * col + 3]
            cax.axis("on")
            cax.imshow(np.linspace(-1, 1, 64)[:, None], aspect="auto", cmap=cmap, origin="lower",
                       extent=[0, 1, -vmax, vmax])
            cax.set_xticks([])
            cax.yaxis.tick_right()
            cax.tick_params(labelsize=7, colors=INK_SECONDARY)
        fig.subplots_adjust(left=0.02, right=0.96, top=1 - 0.9 / (2.1 * nrows + 0.9), bottom=0.02,
                            wspace=0.1, hspace=0.35)
        fig.suptitle(f"{s.label}  |  spatially reliable cells (odd vs even trials), lag {lag}  |  mean z per location; "
                     f"hatched = < min frames; boxes = A/B/C goals  |  page {pg + 1}/{len(pages)}",
                     fontsize=10, x=0.01, ha="left", color=INK_PRIMARY)
        fig.savefig(f"{out_stem}_{pg + 1}.png", dpi=120, facecolor=fig.get_facecolor())
        import matplotlib.pyplot as plt
        plt.close(fig)
    return len(sig)


def plot_within_summary(rows, results, lags, alpha, out_path, root):
    labels_ = list(dict.fromkeys(r["session"] for r in rows))
    fig, axes = new_figure(1, 2, figsize=(14, 5))
    for ax in axes:
        style_axis(ax)
    ax = axes[0]
    w = 0.8 / len(lags)
    ramp = colormap()
    for i, lag in enumerate(lags):
        f = [next(r["frac_reliable"] for r in rows if r["session"] == lb and r["lag"] == lag) for lb in labels_]
        ax.bar(np.arange(len(labels_)) + (i - (len(lags) - 1) / 2) * w, f, width=w,
               color=ramp(0.35 + 0.65 * i / max(len(lags) - 1, 1)), label=f"lag {lag}")
    ax.axhline(alpha, color=INK_MUTED, linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(range(len(labels_)))
    ax.set_xticklabels([lb.split("/")[-1] for lb in labels_], rotation=30, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    title(ax, f"Spatially reliable cells (odd vs even maps, p < {alpha})")
    labels(ax, y="fraction of cells")
    ax = axes[1]
    bins = np.linspace(-1, 1, 41)
    lag = lags[0]
    rs = np.concatenate([results[(lb, lag)]["r"] for lb in labels_])
    nl = np.concatenate([results[(lb, lag)]["null"].ravel() for lb in labels_])
    ax.hist(nl[np.isfinite(nl)], bins=bins, density=True, color=INK_MUTED, alpha=0.5, label="circular-shift null")
    ax.hist(rs[np.isfinite(rs)], bins=bins, density=True, histtype="step", color=SERIES_1, linewidth=2,
            label="cells")
    ax.legend(frameon=False, fontsize=8)
    title(ax, f"Odd/even map correlation, all sessions, lag {lag}")
    labels(ax, "r", "density")
    save(fig, out_path, f"Within-session spatial reliability  |  {root}")


def run_within(sessions, inputs, a, out):
    rows, cells, results = [], [], {}
    (out / "within").mkdir(exist_ok=True)
    for s, (x, loc, trial, ids) in zip(sessions, inputs):
        min_shift = int(30 * s.fs_measured)
        for lag in a.lags:
            r = within_session(x, loc, trial, lag, a.min_frames, a.n_shuffles, min_shift)
            results[(s.label, lag)] = r
            n_sig = plot_within_cells(s, r, ids, lag, a.alpha, a.max_cells,
                                      out / "within" / f"{safe(s.label)}_lag{lag}")
            frac = float(np.mean(r["p"] < a.alpha))
            rows.append({"session": s.label, "lag": lag, "n_cells": x.shape[1], "n_locations": r["n_locs"],
                         "frac_reliable": frac, "median_r": float(np.nanmedian(r["r"])),
                         "null_median_r": float(np.nanmedian(r["null"]))})
            cells.append(pd.DataFrame({"session": s.label, "lag": lag, "cell": ids, "r": r["r"], "p": r["p"]}))
            print(f"{s.label} lag {lag}: {frac:.1%} of {x.shape[1]} cells spatially reliable "
                  f"({r['n_locs']} locations), drew {n_sig}", flush=True)
    pd.DataFrame(rows).to_csv(out / "within_summary.csv", index=False)
    pd.concat(cells).to_csv(out / "within_cells.csv", index=False)
    plot_within_summary(rows, results, a.lags, a.alpha, out / "within_summary.png", a.root)


def main():
    p = parser(__doc__, "spatial_qc")
    p.add_argument("--lags", type=int, nargs="+",
                   help="behaviour at t paired with activity at t + lag (frames; default 0 -500 for "
                        "processed sessions, 0 for raw ones)")
    p.add_argument("--regress", choices=("background", "none"), default="background")
    p.add_argument("--min-frames", type=int, default=20, help="min frames per location (per half for "
                                                              "the within-session test; default 20)")
    p.add_argument("--n-shuffles", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-cells", type=int, default=40, help="max cells drawn per lag and test (default 40)")
    p.add_argument("--tracking", help="raw roots: track_position output folder (default "
                                      "~/Desktop/maze_mini_qc/<name of root>/tracking)")
    a = p.parse_args()
    root, out, sessions = setup(a, "spatial_qc")
    raw = is_raw_root(root)
    if a.lags is None:
        a.lags = [0] if raw else [0, -500]
    if raw:
        tracking = os.path.expanduser(a.tracking or default_out_dir(root, "tracking"))
        inputs = []
        for s in sessions:
            x, loc, trial = raw_inputs(s, a.regress, tracking)
            inputs.append((x, loc, trial, np.flatnonzero(s.cells)))
        run_within(sessions, inputs, a, out)
        print(f"Saved figures and tables to {out}")
        return
    xs = [session_activity(s, a.regress) for s in sessions]
    run_within(sessions, [processed_inputs(s, x) + (np.flatnonzero(s.present),) for s, x in zip(sessions, xs)],
               a, out)
    present = [s.present for s in sessions]
    results, cell_rows, pair_rows = {}, [], []
    for lag in a.lags:
        maps = [spatial_maps(s, x, lag, a.min_frames)[0] for s, x in zip(sessions, xs)]
        cons, null, n, pairs = consistency(maps, present, a.n_shuffles)
        with np.errstate(invalid="ignore"):
            pval = (1 + np.sum(null >= cons, axis=0)) / (1 + a.n_shuffles)
        pval[n < 1] = np.nan
        results[lag] = {"cons": cons, "null": null, "n": n, "p": pval, "pairs": pairs}
        n_sig = plot_cells(sessions, maps, present, results[lag], lag, a.alpha, a.max_cells, out)
        ok = n >= 1
        print(f"lag {lag}: {ok.sum()} cells in >= 2 sessions, {np.mean(pval[ok] < a.alpha):.1%} consistent "
              f"(p < {a.alpha}); mean r {np.nanmean(cons[ok]):.3f} vs null {np.nanmean(null[:, ok]):.3f}; "
              f"drew {n_sig}", flush=True)
        for c in np.flatnonzero(ok):
            cell_rows.append({"lag": lag, "cell": c, "n_sessions": int(sum(pr[c] for pr in present)), "n_pairs": int(n[c]), "consistency": cons[c],
                              "null_mean": float(null[:, c].mean()), "p": pval[c]})
        for q in pairs:
            pair_rows.append({"lag": lag, "a": sessions[q["a"]].label, "b": sessions[q["b"]].label,
                              **{k: v for k, v in q.items() if k not in ("a", "b")}})
    pd.DataFrame(cell_rows).to_csv(out / "cells.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "pairs.csv", index=False)
    plot_summary(sessions, results, a.lags, a.alpha, out / "summary.png", root)
    print(f"Saved figures and tables to {out}")


if __name__ == "__main__":
    main()
