"""
ROI checks for the ABC-task data: do the ROIs look like cells on the mean
images, does every ROICaT column of the aligned activity correspond to one
suite2p ROI, and are cells tracked across sessions in consistent places?

    uv run python quality_control/abc_roi_qc.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of root>/roi_qc):

    fov/<session>.png   chan1 (functional) and chan2 mean images with ROI
                        outlines: blue = matched to a ROICaT column, orange = not
                        in ROICaT (Session.roi_match: mutual best correlation
                        >= --r-min between the suite2p dff and the aligned
                        activity at the behaviour-derived offset)
    summary.png         per session: best-match r per ROI, ROI radius, number of
                        ROIs / ROICaT cells; ROICaT tracking: number of sessions
                        per cell, cells shared by each session pair, and the
                        spread of a tracked cell's position across sessions after
                        removing the median FOV shift of that session pair
                        (a few px = consistent identity; tens of px = suspect)
    roi_match.csv       per suite2p ROI: best ROICaT column, r, matched flag,
                        position, radius, source channel
    roi_summary.csv     per session

Inputs are opened read-only.
"""
import itertools

import numpy as np
import pandas as pd

from abc_common import labels, new_figure, parser, safe, save, setup, title
from suite2p_io import GRIDLINE, INK_MUTED, INK_PRIMARY, SERIES_1, SERIES_2, colormap, style_axis


def outline_image(stat, keep, shape):
    """Boolean outline image of the ROIs in stat[keep]."""
    lab = np.zeros(shape, dtype=np.int32)
    for i in np.flatnonzero(keep):
        lab[stat[i]["ypix"], stat[i]["xpix"]] = i + 1
    edge = np.zeros(shape, bool)
    edge[:-1] |= lab[:-1] != lab[1:]
    edge[1:] |= lab[:-1] != lab[1:]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    edge[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return edge & (lab > 0)


def _show(ax, img, clip=(1, 99.5)):
    lo, hi = np.percentile(img, clip)
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_fov(s, match, matched, out_path):
    fig, axes = new_figure(1, 2, figsize=(16, 8.4))
    stat = s.stat
    for ax, chan in zip(axes, (1, 2)):
        img = s.mean_image(chan)
        _show(ax, img)
        for keep, c in ((matched, SERIES_1), (~matched, SERIES_2)):
            e = outline_image(stat, keep, img.shape)
            rgba = np.zeros(img.shape + (4,))
            rgba[e] = matplotlib_rgba(c)
            ax.imshow(rgba, interpolation="nearest")
        title(ax, f"chan{chan} mean image ({'functional' if chan == 1 else 'structural'})")
    axes[0].plot([], [], "s", color=SERIES_1, label=f"in ROICaT ({matched.sum()})")
    axes[0].plot([], [], "s", color=SERIES_2, label=f"not in ROICaT ({(~matched).sum()})")
    axes[0].legend(frameon=True, fontsize=9, loc="lower left")
    save(fig, out_path, f"{s.label}  |  {len(stat)} ROIs, median match r {np.median(match.best_r[matched]):.3f}")


def matplotlib_rgba(c):
    from matplotlib.colors import to_rgba
    return to_rgba(c)


def tracking_consistency(tables):
    """Per session pair: shared cells, median FOV shift, median residual position
    difference of shared cells after removing that shift (px)."""
    pos = {}
    for t in tables:
        m = t[t.matched]
        pos[t.session.iloc[0]] = m.set_index("roicat_col")[["med_y", "med_x"]]
    rows = []
    for a, b in itertools.combinations(pos, 2):
        shared = pos[a].index.intersection(pos[b].index)
        if len(shared) < 3:
            rows.append({"a": a, "b": b, "shared": len(shared)})
            continue
        d = pos[b].loc[shared].to_numpy() - pos[a].loc[shared].to_numpy()
        shift = np.median(d, axis=0)
        res = np.linalg.norm(d - shift, axis=1)
        rows.append({"a": a, "b": b, "shared": len(shared), "shift_y": shift[0], "shift_x": shift[1],
                     "median_residual_px": float(np.median(res)),
                     "frac_residual_gt_10px": float(np.mean(res > 10))})
    return pd.DataFrame(rows)


def plot_summary(per, tables, pairs, presence, out_path, root, r_min):
    names = [l.split("/")[-1] for l in per.session]
    x = np.arange(len(names))
    fig, axes = new_figure(2, 3, figsize=(19, 10.5))
    a = axes.ravel()
    for ax in a:
        style_axis(ax)
    rng = np.random.default_rng(0)

    for ax, col, ylab, ttl in ((a[0], "best_r", "r (suite2p dff vs ROICaT column)", "Best ROICaT match per ROI"),
                               (a[1], "radius", "radius (px)", "ROI radius")):
        for i, t in enumerate(tables):
            for keep, c, lab in ((t.matched, SERIES_1, "in ROICaT"), (~t.matched, SERIES_2, "not in ROICaT")):
                v = t.loc[keep, col].to_numpy()
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=6, color=c, alpha=0.5, edgecolor="none",
                           label=lab if i == 0 else None, zorder=2)
            med = np.median(t[col])
            ax.plot([i - 0.3, i + 0.3], [med, med], color=INK_PRIMARY, linewidth=2, zorder=3)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.legend(frameon=False, fontsize=8, markerscale=2)
        title(ax, ttl)
        labels(ax, y=ylab)
    a[0].axhline(r_min, color=INK_MUTED, linestyle="--", linewidth=1)

    ax = a[2]
    ax.plot(x, per.n_roi, "o-", color=SERIES_1, markersize=7, label="suite2p ROIs", markeredgecolor="white",
            markeredgewidth=1.5, zorder=3)
    ax.plot(x, per.roicat_present, "o-", color=SERIES_2, markersize=7, label="ROICaT cells",
            markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8)
    title(ax, "ROIs per session")
    labels(ax, y="count")
    for ax in a[:3]:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)

    ax = a[3]
    n_sess = presence.sum(axis=0)
    n_sess = n_sess[n_sess > 0]
    vals, cnt = np.unique(n_sess, return_counts=True)
    ax.bar(vals, cnt, color=SERIES_1, edgecolor="white")
    ax.set_xticks(range(1, len(names) + 1))
    title(ax, f"ROICaT cells by number of sessions ({len(n_sess)} cells)")
    labels(ax, "sessions the cell is present in", "cells")

    ax = a[4]
    S = presence.astype(int) @ presence.T.astype(int)
    im = ax.imshow(S, cmap=colormap())
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, S[i, j], ha="center", va="center", fontsize=8,
                    color="white" if S[i, j] > 0.6 * S.max() else INK_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax, label="shared cells")
    title(ax, "ROICaT cells shared by session pairs")

    ax = a[5]
    R = np.full((len(names), len(names)), np.nan)
    idx = {l: i for i, l in enumerate(per.session)}
    for _, r in pairs.iterrows():
        if "median_residual_px" in r and np.isfinite(r.get("median_residual_px", np.nan)):
            R[idx[r.a], idx[r.b]] = R[idx[r.b], idx[r.a]] = r.median_residual_px
    im = ax.imshow(R, cmap=colormap(), vmin=0)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(R[i, j]):
                ax.text(j, i, f"{R[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if R[i, j] > 0.6 * np.nanmax(R) else INK_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax, label="median residual (px)")
    title(ax, "Tracked-cell position mismatch after FOV shift")
    save(fig, out_path, f"ROIs and ROICaT tracking  |  {root}")


def main():
    p = parser(__doc__, "roi_qc")
    p.add_argument("--r-min", type=float, default=0.8, help="min correlation for a ROI-column match (default 0.8)")
    a = p.parse_args()
    root, out, sessions = setup(a, "roi_qc")
    (out / "fov").mkdir(exist_ok=True)
    tables, per = [], []
    presence = np.array([s.present for s in sessions])
    for s in sessions:
        m = s.roi_match()
        matched = m.matched(a.r_min)
        rt = s.roi_table
        t = pd.DataFrame({"session": s.label, "roi": np.arange(len(m.best_r)), "roicat_col": m.best_col,
                          "best_r": m.best_r, "mutual": m.mutual, "matched": matched,
                          "med_y": rt.med_y, "med_x": rt.med_x, "radius": rt.radius, "npix": rt.npix,
                          "source": rt.roi_source_channel})
        tables.append(t)
        cols_unmatched = len(set(m.columns) - set(m.best_col[matched]))
        per.append({"session": s.label, "n_roi": len(t), "roicat_present": int(s.present.sum()),
                    "matched": int(matched.sum()), "roicat_cols_without_roi": cols_unmatched,
                    "median_match_r": float(np.median(m.best_r[matched])),
                    "min_match_r": float(m.best_r[matched].min()), "median_radius": float(rt.radius.median())})
        plot_fov(s, m, matched, out / "fov" / f"{safe(s.label)}.png")
        print(f"{s.label}: {matched.sum()}/{len(t)} ROIs matched (r >= {a.r_min}), "
              f"{cols_unmatched} ROICaT columns without a ROI, median r {per[-1]['median_match_r']:.3f}")
    per = pd.DataFrame(per)
    pairs = tracking_consistency(tables)
    print(pairs.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    pd.concat(tables, ignore_index=True).to_csv(out / "roi_match.csv", index=False)
    per.to_csv(out / "roi_summary.csv", index=False)
    pairs.to_csv(out / "tracking_pairs.csv", index=False)
    plot_summary(per, tables, pairs, presence, out / "summary.png", root, a.r_min)
    print(f"Saved figures and tables to {out}")


if __name__ == "__main__":
    main()
