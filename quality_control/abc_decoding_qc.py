"""
Population decoding sanity checks for the ABC-task data: can maze location and
goal progress be read out of the aligned activity better than chance, within a
session and across sessions (using the cells ROICaT tracked in both)?

    uv run python quality_control/abc_decoding_qc.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding

Features: aligned ROICaT activity of the cells present, z-scored per cell and
smoothed with a --smooth-frame running mean (default 3 frames, ~0.4 s).
--regress background|pop removes a global signal from each z-scored trace
first (least squares, as in abc_raw_tuning_qc): the mean suite2p Fneu over all
ROIs, or the population mean of the z-scored activity (default none). The
shared signal is poke-locked here, so by itself it carries some goal progress
(and, through the goals, location); with it regressed out, what is left is
decoding from the differences between cells.
Control decoder (always run): the two global signals alone (background and
population mean, z-scored, smoothed) as features, i.e. how far a shared trace
without any single-cell information gets.
Decoder: shrinkage LDA (sklearn, solver lsqr, shrinkage auto). Score: balanced
accuracy (chance = 1 / n classes, robust to unequal occupancy).

Targets
    location        maze node 1-8 of each frame (edges and untracked frames left out);
                    --exclude-goals also leaves out the frames at the session's
                    A/B/C nodes (5 classes): place information beyond the goal
                    ports, which dominate the location confusion otherwise
    goal_progress   time from one reward (A/B/C_on) to the next in --gp-bins equal
                    bins (default 10; Session.goal_progress): where the animal is
                    between two goals, whichever goals they are. Frames before
                    the first and after the last reward are left out.

Within session: cross-validated over --folds contiguous blocks of trials, so
training and test frames never come from the same trial.
Across sessions: train on all frames of session i and test on session j, using
only the cells present in both (pairs with < --min-shared cells are skipped).
Goal progress keeps its meaning across sessions although the A/B/C nodes move;
location is the physical node.

--balance-train [frames|visits] balances the training sets (random, fixed
seed; test sets untouched). frames (the default when given without a value):
every class subsampled to the size of the smallest one, so the decoder cannot
favour the locations the animal occupies most (the class-frequency prior of
LDA). visits: every class gets the same number of visits (contiguous runs of
the label; the rarest class's count), at most --visit-frames frames from each
(default 5, ~0.7 s), then frames equalised. Tried against the pull of errors
onto goal nodes in some processed sessions (CLAUDE.md): it does not reduce it
and costs training data (lower accuracy, weaker cross-session), so frames is
the recommended mode. Goal pull (decoding.csv, printed): share of errors on
non-goal-node frames that land on a goal node (3/7 if errors were spread
evenly over 8 nodes). Evenness (decoding.csv, printed for location): SD across
classes of the recall (confusion-matrix diagonal) and normalised entropy of
the predicted classes (1 = every class predicted equally often).

--average-bins: goal progress decoded from activity averaged within each bin
of each segment (one sample per segment x bin; segments found where the bin
drops), i.e. closer to how tuning curves are built. Single frames carry little
of the weak per-cell tuning (CLAUDE.md); averaging is what makes it visible.
Cross-session decoding stays frame-wise.
--max-excess-steps N: goal progress only from near-direct segments, i.e.
reward-to-reward segments whose route (distinct nodes visited, from the
location labels) is at most N node steps longer than the shortest path
between the two goals (segment_route_stats). Long detours stretch the bins over
very different behaviour. Keeps ~45-90 % of segments, 25-80 % of frames at 3.
--max-wrong-pokes N: likewise, segments with at most N pokes at ports other
than the two goals (pyControl). Both can be combined (a segment must pass
both); excess steps and wrong pokes are strongly correlated (rho 0.5-0.9).
Circular error (goal progress, always): mean circular distance between true
and predicted bin as a fraction of a cycle (0.25 = random); decoding.csv has
it with the mean and 5th percentile of its label-shift null.

Goal share: fraction of location frames at the A/B/C nodes vs fraction of
predictions on them (decoding.csv, printed). Predictions piling onto the goal
nodes mean the decoder largely tells "at a goal port", not each node. Goal
nodes are labelled A/B/C (goal colours) in the location confusion matrices.

Null: the same pipeline with the labels circularly shifted against the
activity (>= 30 s; --n-shuffles), which keeps the temporal structure of both.
Behaviour also carries goal progress (the animal runs, then slows at the
port), so above-chance decoding can partly reflect speed-related activity;
and goal progress and location are not independent in a given task.

Lag scan (default on): within-session accuracy with the labels at t decoded
from activity at t - lag, for lags over +-90 s. With correct alignment it
peaks at lag 0. In the AB_08 sessions it is at chance at 0 and peaks at
~+480-510 frames (~65 s) in all six sessions; --lag 490 reruns everything at
that lag. See CLAUDE.md (known findings).

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of root>/decoding_qc):
    summary.png    within-session accuracy vs null per target (cells and
                   global-signals-only control); confusion
                   matrices (location, goal progress) for the session with most trials;
                   cross-session accuracy matrices (minus the null mean)
    lag_scan.png   accuracy vs lag per session and target (dashed: global
                   signals only)
    decoding.csv, cross_session.csv, lag_scan.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from abc_common import (GOAL_COLORS, direct_only, labels, new_figure, parser, save, segment_route_stats, setup,
                        title)
from abc_data import maze
from abc_raw_tuning_qc import regress_out
from suite2p_io import GRIDLINE, INK_MUTED, INK_PRIMARY, SERIES_1, SERIES_2, colormap, style_axis

warnings.filterwarnings("ignore", category=UserWarning)

TARGETS = ("location", "goal_progress")
N_GP_BINS = 10  # set from --gp-bins
REGRESS = "none"  # set from --regress
EXCLUDE_GOALS = False  # set from --exclude-goals
BALANCE = "none"  # set from --balance-train: none | frames | visits
VISIT_FRAMES = 5  # set from --visit-frames
AVERAGE_BINS = False  # set from --average-bins
MAX_EXCESS = None  # set from --max-excess-steps
MAX_POKES = None  # set from --max-wrong-pokes


def zscore(x):
    return np.nan_to_num((x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0))


def smoothed(x, smooth):
    if smooth > 1:
        k = np.ones(smooth) / smooth
        x = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, x)
    return x


def global_signal(s, kind):
    """(T,) 'background' = mean suite2p Fneu over all ROIs at the aligned frames,
    'pop' = mean of the z-scored activity of the cells present."""
    if kind == "background":
        g = np.nanmean(s.suite2p_dff_aligned("Fneu"), axis=1)
        return np.where(np.isfinite(g), g, np.nanmean(g))
    if kind == "pop":
        return zscore(np.asarray(s.activity, dtype=np.float64)[:, s.present]).mean(axis=1)
    raise ValueError(kind)


def features(s, cols, smooth):
    x = zscore(np.asarray(s.activity, dtype=np.float64)[:, cols])
    if REGRESS != "none":
        x = zscore(regress_out(x, global_signal(s, REGRESS))[0])
    return smoothed(x, smooth)


def global_features(s, smooth):
    """(T, 2) control features: background and population mean only."""
    return smoothed(zscore(np.c_[global_signal(s, "background"), global_signal(s, "pop")]), smooth)


def location_labels(loc, task, exclude_goals=None):
    """Node 1-8 per frame, -1 for edges, node 9, untracked frames and (with
    exclude_goals, default --exclude-goals) the A/B/C nodes in task."""
    loc = np.where(np.isfinite(loc), loc, -1).astype(int)
    ok = maze.is_node(loc) & (loc != 9)
    if EXCLUDE_GOALS if exclude_goals is None else exclude_goals:
        ok &= ~np.isin(loc, task)
    return np.where(ok, loc, -1)


def goal_pull(y, pred, task):
    """Share of errors on non-goal-node frames that land on a goal node."""
    err = ~np.isin(y, task) & (pred != y)
    return float(np.mean(np.isin(pred[err], task))) if err.any() else np.nan


def goal_share(y, pred, task):
    """Fraction of location frames at the goal nodes, and of predictions on them."""
    return float(np.mean(np.isin(y, task))), float(np.mean(np.isin(pred, task)))


def targets(s):
    gp = s.goal_progress(N_GP_BINS)
    if MAX_EXCESS is not None or MAX_POKES is not None:
        gp = direct_only(gp, segment_route_stats(s, s.location), MAX_EXCESS, MAX_POKES)
    return {"location": location_labels(s.location, s.task), "goal_progress": gp}


def trial_folds(s, n_folds):
    """(T,) fold id per frame from contiguous blocks of trials (-1 = no trial)."""
    tr = np.full(s.n_frames, -1)
    for i, (a, *_, end) in enumerate(s.trial_times):
        tr[a:end] = i
    n = len(s.trial_times)
    edges = np.linspace(0, n, n_folds + 1).round().astype(int)
    fold = np.full(s.n_frames, -1)
    for f in range(n_folds):
        fold[(tr >= edges[f]) & (tr < edges[f + 1])] = f
    return fold


def _lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def visit_ids(y):
    """(T,) id of the contiguous run of equal labels each frame belongs to."""
    return np.cumsum(np.r_[True, y[1:] != y[:-1]])


def balance_mode(balance=None):
    b = BALANCE if balance is None else balance
    return {True: "frames", False: "none", None: "none"}.get(b, b)


def fit(x, y, balance=None, seed=0, visits=None):
    """LDA fitted on (x, y). balance (default --balance-train): 'frames' = every
    class randomly subsampled to the size of the smallest one; 'visits' = the
    same number of visits per class (visits: run id per sample, visit_ids),
    at most VISIT_FRAMES frames each, then frames equalised."""
    mode = balance_mode(balance)
    rng = np.random.default_rng(seed)
    if mode == "visits" and visits is not None:
        classes = np.unique(y)
        per = {c: np.unique(visits[y == c]) for c in classes}
        n_vis = min(len(v) for v in per.values())
        keep = []
        for c in classes:
            for v in rng.choice(per[c], n_vis, replace=False):
                idx = np.flatnonzero((visits == v) & (y == c))
                keep.append(rng.choice(idx, min(VISIT_FRAMES, len(idx)), replace=False))
        keep = np.concatenate(keep)
        x, y = x[keep], y[keep]
    if mode in ("frames", "visits"):
        classes, counts = np.unique(y, return_counts=True)
        keep = np.concatenate([rng.choice(np.flatnonzero(y == c), counts.min(), replace=False) for c in classes])
        x, y = x[keep], y[keep]
    return _lda().fit(x, y)


def evenness(y, pred, classes):
    """SD across classes of the recall, and normalised entropy of the predicted
    class distribution (1 = all classes predicted equally often)."""
    recall = np.array([np.mean(pred[y == c] == c) for c in classes if (y == c).any()])
    q = np.array([np.mean(pred == c) for c in classes])
    q = q[q > 0]
    return float(np.std(recall)), float(-(q * np.log(q)).sum() / np.log(len(classes)))


def segment_ids(gp):
    """(T,) running segment number from goal-progress labels (a new segment
    starts where the bin drops or labels resume), -1 outside."""
    ok = gp >= 0
    prev_ok = np.r_[False, ok[:-1]]
    start = ok & (~prev_ok | np.r_[True, gp[1:] < gp[:-1]])
    return np.where(ok, np.cumsum(start) - 1, -1)


def bin_means(x, y, fold):
    """One sample per (segment, bin): mean of x over its frames, its bin, its fold."""
    seg = segment_ids(y)
    ok = (seg >= 0) & (fold >= 0)
    key = seg[ok] * (y.max() + 1) + y[ok]
    ks, inv = np.unique(key, return_inverse=True)
    X = np.zeros((len(ks), x.shape[1]))
    np.add.at(X, inv, x[ok])
    X /= np.bincount(inv)[:, None]
    F = np.zeros(len(ks), dtype=int)
    F[inv] = fold[ok]
    return X, ks % (y.max() + 1), F


def circular_error(y, pred, n_bins):
    """Mean circular distance between true and predicted bin, fraction of a cycle."""
    if len(y) == 0:
        return np.nan
    d = np.abs(np.asarray(y) - np.asarray(pred))
    return float(np.mean(np.minimum(d, n_bins - d)) / n_bins)


def cv_decode(x, y, fold, balance=None, average=False):
    """Balanced accuracy and predictions of cross-validated LDA; average: on
    segment x bin means (goal progress) instead of frames."""
    if average:
        x, y, fold = bin_means(x, y, fold)
    vis = visit_ids(y)
    ok = (y >= 0) & (fold >= 0)
    pred = np.full(len(y), -1)
    for f in np.unique(fold[ok]):
        tr, te = ok & (fold != f), ok & (fold == f)
        cnt = np.unique(y[tr], return_counts=True)[1]
        balanced = balance_mode(balance) != "none"
        if len(cnt) < 2 or not te.any() or cnt.sum() <= len(cnt) or (balanced and cnt.min() < 2):
            continue   # too few training samples (e.g. 08-06, 2 trials, with --average-bins)
        pred[te] = fit(x[tr], y[tr], balance, seed=int(f), visits=vis[tr]).predict(x[te])
    m = ok & (pred >= 0)
    if not m.any():   # e.g. too few trials for the folds
        return np.nan, y[m], pred[m]
    return balanced_accuracy_score(y[m], pred[m]), y[m], pred[m]


def apply_lag(x, y, lag):
    """Pair label y[t] with activity x[t - lag] (lag > 0: activity from earlier).
    Not circular: frames without a partner get label -1."""
    if lag == 0:
        return x, y
    xs = np.roll(x, lag, axis=0)
    t = np.arange(len(y))
    valid = (t >= lag) if lag > 0 else (t < len(y) + lag)
    return xs, np.where(valid, y, -1)


def lag_scan(s, lags, n_folds, smooth, x=None):
    """Balanced accuracy per target and lag (frames); x: features (default cells)."""
    if x is None:
        x = features(s, np.flatnonzero(s.present), smooth)
    fold = trial_folds(s, n_folds)
    return {name: np.array([cv_decode(*apply_lag(x, y, L), fold,
                                      average=AVERAGE_BINS and name == "goal_progress")[0] for L in lags])
            for name, y in targets(s).items()}


def shifted(y, rng, min_shift):
    return np.roll(y, rng.integers(min_shift, len(y) - min_shift))


def within_session(s, n_folds, n_shuffles, smooth, lag=0, seed=0, x=None):
    """x: features (default the cells present)."""
    if x is None:
        x = features(s, np.flatnonzero(s.present), smooth)
    fold = trial_folds(s, n_folds)
    rng = np.random.default_rng(seed)
    min_shift = int(30 * s.fs_measured)
    out = {}
    for name, y in targets(s).items():
        x_, y = apply_lag(x, y, lag)
        avg = AVERAGE_BINS and name == "goal_progress"
        acc, yt, yp = cv_decode(x_, y, fold, average=avg)
        null, null_circ = [], []
        for _ in range(n_shuffles):
            a_, yt_, yp_ = cv_decode(x_, shifted(y, rng, min_shift), fold, average=avg)
            null.append(a_)
            null_circ.append(circular_error(yt_, yp_, N_GP_BINS))
        out[name] = {"acc": acc, "null": np.array(null), "y": yt, "pred": yp,
                     "classes": np.unique(y[y >= 0]), "chance": 1 / len(np.unique(y[y >= 0]))}
        if name == "goal_progress":
            out[name].update(circ=circular_error(yt, yp, N_GP_BINS), null_circ=np.array(null_circ))
    return out


def cross_session(sessions, n_shuffles, smooth, min_shared, lag=0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, si in enumerate(sessions):
        for j, sj in enumerate(sessions):
            if i == j:
                continue
            shared = np.flatnonzero(si.present & sj.present)
            if len(shared) < min_shared:
                continue
            xi, xj = features(si, shared, smooth), features(sj, shared, smooth)
            ti, tj = targets(si), targets(sj)
            for name in TARGETS:
                xi_, yi = apply_lag(xi, ti[name], lag)
                xj_, yj = apply_lag(xj, tj[name], lag)
                common = np.intersect1d(np.unique(yi[yi >= 0]), np.unique(yj[yj >= 0]))
                mi, mj = np.isin(yi, common), np.isin(yj, common)
                model = fit(xi_[mi], yi[mi], visits=visit_ids(yi)[mi])
                pred = model.predict(xj_)
                acc = balanced_accuracy_score(yj[mj], pred[mj])
                null = []
                for _ in range(n_shuffles):
                    ys = shifted(yj, rng, int(30 * sj.fs_measured))
                    ms = np.isin(ys, common)
                    null.append(balanced_accuracy_score(ys[ms], pred[ms]))
                rows.append({"train": si.label, "test": sj.label, "target": name, "n_shared": len(shared),
                             "n_classes": len(common), "acc": acc, "null_mean": float(np.mean(null)),
                             "null_p95": float(np.percentile(null, 95)),
                             "p": float((np.sum(np.array(null) >= acc) + 1) / (len(null) + 1))})
    return pd.DataFrame(rows)


def plot_accuracy(ax, names, within, within_glob, tgt):
    """Within-session accuracy per session (dots) vs label-shift null, with the
    global-signals-only control. within: list of within_session() results."""
    rng = np.random.default_rng(0)
    for i, w in enumerate(within):
        r = w[tgt]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(r["null"])), r["null"], s=8, color=INK_MUTED, alpha=0.5,
                   edgecolor="none", label="label-shift null" if i == 0 else None, zorder=2)
        ax.plot(i, r["acc"], "o", color=SERIES_1, markersize=10, markeredgecolor="white", markeredgewidth=1.5,
                label="decoder (cells)" if i == 0 else None, zorder=3)
        ax.plot(i + 0.25, within_glob[i][tgt]["acc"], "D", color=SERIES_2, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2,
                label="global signals only" if i == 0 else None, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [r["chance"]] * 2, color=INK_PRIMARY, linewidth=1, linestyle="--",
                label="chance" if i == 0 else None)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8)
    title(ax, f"Within-session decoding of {tgt.replace('_', ' ')}")
    labels(ax, y="balanced accuracy")


def plot_confusion(fig, ax, r, tgt, heading, goals=None):
    """Row-normalised confusion matrix of one within_session() target result;
    goals: (3,) A/B/C nodes, labelled in goal colours on a location matrix."""
    C = confusion_matrix(r["y"], r["pred"], labels=r["classes"], normalize="true")
    im = ax.imshow(C, cmap=colormap(), vmin=0, vmax=1)
    goals = list(goals) if (goals is not None and tgt == "location") else []
    ticks = [f"{c} {'ABC'[goals.index(c)]}" if c in goals else str(c) for c in r["classes"]] \
        if tgt == "location" else [str(c + 1) for c in r["classes"]]
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_yticks(range(len(ticks)))
    ax.set_yticklabels(ticks, fontsize=8)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        c = lab.get_text().split(" ")[-1]
        if c in ("A", "B", "C"):
            lab.set_color(GOAL_COLORS["ABC".index(c)])
            lab.set_fontweight("bold")
    for i in range(len(ticks)):
        for j in range(len(ticks)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if C[i, j] > 0.6 else INK_PRIMARY)
    fig.colorbar(im, ax=ax, label="fraction of true class")
    title(ax, f"{tgt.replace('_', ' ')} confusion, {heading}")
    labels(ax, "predicted" + (" bin" if tgt != "location" else " node"), "true")


def plot_summary(sessions, within, within_glob, cross, out_path, root):
    names = [s.label.split("/")[-1] for s in sessions]
    fig, axes = new_figure(2, 3, figsize=(20, 11.5))
    a = axes.ravel()
    for ax in a:
        style_axis(ax)
    for ax, tgt in zip((a[0], a[3]), TARGETS):
        plot_accuracy(ax, names, within, within_glob, tgt)
    best = int(np.argmax([len(s.trial_times) for s in sessions]))
    for ax, tgt in zip((a[1], a[4]), TARGETS):
        plot_confusion(fig, ax, within[best][tgt], tgt,
                       f"{names[best]} (task {'-'.join(map(str, sessions[best].task))})", sessions[best].task)

    for ax, tgt in zip((a[2], a[5]), TARGETS):
        D = np.full((len(names), len(names)), np.nan)
        P = np.full_like(D, np.nan)
        idx = {s.label: i for i, s in enumerate(sessions)}
        for _, r in cross[cross.target == tgt].iterrows():
            D[idx[r.train], idx[r.test]] = r.acc - r.null_mean
            P[idx[r.train], idx[r.test]] = r.p
        for i, w in enumerate(within):
            D[i, i] = w[tgt]["acc"] - np.mean(w[tgt]["null"])
        lim = np.nanmax(np.abs(D)) if np.isfinite(D).any() else 1
        im = ax.imshow(D, cmap=colormap("diverging"), vmin=-lim, vmax=lim)
        for i in range(len(names)):
            for j in range(len(names)):
                if np.isfinite(D[i, j]):
                    star = "*" if (i != j and P[i, j] < 0.05) else ""
                    ax.text(j, i, f"{D[i, j]:+.2f}{star}", ha="center", va="center", fontsize=7.5,
                            color="white" if abs(D[i, j]) > 0.6 * lim else INK_PRIMARY)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        fig.colorbar(im, ax=ax, label="accuracy − null mean")
        title(ax, f"Cross-session {tgt.replace('_', ' ')} (rows train, cols test; * p<0.05; diag = within)")
        labels(ax, "test session", "train session")
    save(fig, out_path, f"Population decoding  |  {root}")


def plot_lag_scan(sessions, lags, scans, scans_glob, within, out_path, root):
    fig, axes = new_figure(1, 2, figsize=(17, 5.8))
    ramp = colormap()
    for ax, tgt in zip(axes, TARGETS):
        style_axis(ax)
        for i, (s, sc, sg) in enumerate(zip(sessions, scans, scans_glob)):
            c = ramp(0.3 + 0.7 * i / max(len(sessions) - 1, 1))
            ax.plot(lags / s.fs_measured, sc[tgt], color=c, linewidth=1.5, label=s.label.split("/")[-1])
            ax.plot(lags / s.fs_measured, sg[tgt], color=c, linewidth=1, linestyle="--",
                    label="global signals only" if i == len(sessions) - 1 else None)
        ax.axhline(np.mean([w[tgt]["chance"] for w in within]), color=INK_PRIMARY, linestyle="--",
                   linewidth=1, label="chance")
        ax.axvline(0, color=INK_MUTED, linewidth=1)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        title(ax, f"{tgt.replace('_', ' ')}: decoding vs activity lag")
        labels(ax, "lag (s): label at t decoded from activity at t − lag", "balanced accuracy")
    save(fig, out_path, f"Lag scan (peak away from 0 = neural data and behaviour offset)  |  {root}")


def main():
    p = parser(__doc__, "decoding_qc")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-shuffles", type=int, default=50)
    p.add_argument("--smooth", type=int, default=3, help="running-mean width in frames (default 3)")
    p.add_argument("--min-shared", type=int, default=30, help="min shared cells for cross-session decoding")
    p.add_argument("--lag", type=int, default=0,
                   help="decode labels at t from activity at t - lag frames (exploration; default 0)")
    p.add_argument("--lag-range", type=float, default=90, help="lag scan range, +-s (default 90; 0 = no scan)")
    p.add_argument("--lag-step", type=int, default=10, help="lag scan step, frames (default 10)")
    p.add_argument("--gp-bins", type=int, default=10, help="goal-progress bins between rewards (default 10)")
    p.add_argument("--regress", choices=("none", "background", "pop"), default="none",
                   help="global signal regressed out of each cell before decoding (default none)")
    p.add_argument("--exclude-goals", action="store_true",
                   help="location: leave out frames at the session's A/B/C nodes (5 classes)")
    p.add_argument("--balance-train", nargs="?", const="frames", default="none",
                   choices=("none", "frames", "visits"),
                   help="balance training classes by frames (default when given) or visits")
    p.add_argument("--visit-frames", type=int, default=5,
                   help="--balance-train visits: max frames per visit (default 5)")
    p.add_argument("--average-bins", action="store_true",
                   help="goal progress: decode segment x bin means instead of frames (within session)")
    p.add_argument("--max-excess-steps", type=int,
                   help="goal progress: only segments at most N node steps longer than the shortest path")
    p.add_argument("--max-wrong-pokes", type=int,
                   help="goal progress: only segments with at most N pokes at non-goal ports")
    a = p.parse_args()
    global N_GP_BINS, REGRESS, EXCLUDE_GOALS, BALANCE, AVERAGE_BINS, VISIT_FRAMES, MAX_EXCESS, MAX_POKES
    N_GP_BINS, REGRESS, EXCLUDE_GOALS, BALANCE, AVERAGE_BINS, VISIT_FRAMES = \
        a.gp_bins, a.regress, a.exclude_goals, a.balance_train, a.average_bins, a.visit_frames
    MAX_EXCESS, MAX_POKES = a.max_excess_steps, a.max_wrong_pokes
    root, out, sessions = setup(a, "decoding_qc")
    within, within_glob, rows = [], [], []
    for s in sessions:
        w = within_session(s, a.folds, a.n_shuffles, a.smooth, a.lag)
        wg = within_session(s, a.folds, a.n_shuffles, a.smooth, a.lag, x=global_features(s, a.smooth))
        within.append(w)
        within_glob.append(wg)
        for feats, res in (("cells", w), ("global", wg)):
            for tgt, r in res.items():
                p_val = (np.sum(r["null"] >= r["acc"]) + 1) / (len(r["null"]) + 1)
                share = goal_share(r["y"], r["pred"], s.task) if tgt == "location" else (np.nan, np.nan)
                ev = evenness(r["y"], r["pred"], r["classes"]) if feats == "cells" else (np.nan, np.nan)
                rows.append({"session": s.label, "target": tgt, "features": feats, "regress": a.regress,
                             "exclude_goals": a.exclude_goals, "balance_train": a.balance_train,
                             "average_bins": a.average_bins, "max_excess_steps": a.max_excess_steps,
                             "max_wrong_pokes": a.max_wrong_pokes,
                             "n_segments": int(len(np.unique(segment_ids(r["y"])))) if tgt == "goal_progress" else np.nan,
                             "circ_err": r.get("circ", np.nan),
                             "null_circ_mean": float(np.mean(r["null_circ"])) if "null_circ" in r else np.nan,
                             "null_circ_p5": float(np.percentile(r["null_circ"], 5)) if "null_circ" in r else np.nan,
                             "recall_sd": ev[0], "pred_evenness": ev[1],
                             "goal_pull": goal_pull(r["y"], r["pred"], s.task) if tgt == "location" else np.nan,
                             "n_cells": int(s.present.sum()), "n_classes": len(r["classes"]), "chance": r["chance"],
                             "acc": r["acc"], "null_mean": float(np.mean(r["null"])),
                             "null_p95": float(np.percentile(r["null"], 95)), "p": p_val,
                             "frac_true_on_goals": share[0], "frac_pred_on_goals": share[1]})
        print(f"{s.label}: location {w['location']['acc']:.3f} (null {np.mean(w['location']['null']):.3f}, "
              f"global only {wg['location']['acc']:.3f}), goal progress {w['goal_progress']['acc']:.3f} "
              f"(null {np.mean(w['goal_progress']['null']):.3f}, global only {wg['goal_progress']['acc']:.3f}, "
              f"circular error {w['goal_progress']['circ']:.3f} vs null {np.mean(w['goal_progress']['null_circ']):.3f})"
              + ("" if a.exclude_goals else " | location frames on goals {:.2f}, predictions on goals {:.2f}".format(
                  *goal_share(w["location"]["y"], w["location"]["pred"], s.task)))
              + " | location recall SD {:.3f}, prediction evenness {:.2f}".format(
                  *evenness(w["location"]["y"], w["location"]["pred"], w["location"]["classes"]))
              + " | goal pull {:.2f} (even 0.43)".format(goal_pull(w["location"]["y"], w["location"]["pred"], s.task)),
              flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "decoding.csv", index=False)
    cross = cross_session(sessions, max(a.n_shuffles, 100), a.smooth, a.min_shared, a.lag)
    cross.to_csv(out / "cross_session.csv", index=False)
    for tgt in TARGETS:
        c = cross[cross.target == tgt]
        print(f"cross-session {tgt}: median acc - null {np.median(c.acc - c.null_mean):+.3f}, "
              f"{np.mean(c.p < 0.05):.0%} of {len(c)} pairs p < 0.05")
    note = (f"  |  activity lag {a.lag} frames" if a.lag else "") + \
        (f"  |  {a.regress} regressed" if a.regress != "none" else "") + \
        ("  |  goal nodes excluded from location" if a.exclude_goals else "") + \
        (f"  |  training balanced by {a.balance_train}" if a.balance_train != "none" else "") + \
        ("  |  goal progress on segment x bin means" if a.average_bins else "") + \
        (f"  |  goal progress segments <= {a.max_excess_steps} extra steps" if a.max_excess_steps is not None else "") + \
        (f"  |  goal progress segments <= {a.max_wrong_pokes} wrong pokes" if a.max_wrong_pokes is not None else "")
    plot_summary(sessions, within, within_glob, cross, out / "summary.png", root + note)
    if a.lag_range > 0:
        n = int(round(a.lag_range * sessions[0].fs_measured / a.lag_step)) * a.lag_step
        lags = np.arange(-n, n + 1, a.lag_step)
        scans, scans_glob = [], []
        for s in sessions:
            scans.append(lag_scan(s, lags, a.folds, a.smooth))
            scans_glob.append(lag_scan(s, lags, a.folds, a.smooth, x=global_features(s, a.smooth)))
            i = {t: int(np.nanargmax(v)) if np.isfinite(v).any() else 0 for t, v in scans[-1].items()}
            print(f"{s.label} lag scan: best location lag {lags[i['location']]} frames "
                  f"({scans[-1]['location'][i['location']]:.3f}), best goal-progress lag "
                  f"{lags[i['goal_progress']]} frames ({scans[-1]['goal_progress'][i['goal_progress']]:.3f})", flush=True)
        pd.DataFrame({"lag_frames": np.tile(lags, len(sessions)),
                      "session": np.repeat([s.label for s in sessions], len(lags)),
                      **{t: np.concatenate([sc[t] for sc in scans]) for t in TARGETS},
                      **{f"{t}_global": np.concatenate([sc[t] for sc in scans_glob]) for t in TARGETS}}
                     ).to_csv(out / "lag_scan.csv", index=False)
        plot_lag_scan(sessions, lags, scans, scans_glob, within, out / "lag_scan.png", root + note)
    print(f"Saved figures and tables to {out}")


if __name__ == "__main__":
    main()
