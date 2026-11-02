"""
Lag check on our own processing of raw ABC sessions: suite2p run on the raw
ScanImage tif, synced to behaviour from the raw DAQ / pyControl signals
(abc_data/raw.py), and goal progress between rewards and maze location decoded
as a function of the lag between activity and behaviour.

    uv run python quality_control/abc_raw_lag_qc.py /Volumes/behrens/Sandra/mini2p/Arya/RAW_GOOD

This is the control for the ~500-frame (~66 s) offset found in Arya's
processed sessions (CLAUDE.md, known findings): with a clean pipeline the
decoding peak should sit at lag 0.

Per session:
    sync        suite2p frames vs DAQ FrameClock edges, rsync counts DAQ /
                pyControl, clock-fit residual
    decoding    goal progress (--gp-bins bins from one reward to the next) and
                location (maze node 1-8 per frame from preprocessing/
                track_position, <session>_imaging_location.npy in --tracking,
                default ~/Desktop/maze_mini_qc/<name of root>/tracking; edges
                and untracked frames left out, as in abc_decoding_qc;
                --exclude-goals also leaves out the A/B/C nodes;
                --balance-train [frames|visits] balances the training classes;
                --average-bins decodes goal progress from segment x bin means;
                --max-excess-steps / --max-wrong-pokes keep only near-direct
                goal-progress segments;
                see abc_decoding_qc) from
                F - 0.7 * Fneu of all suite2p ROIs, z-scored and smoothed over
                --smooth frames; shrinkage LDA, cross-validated over --folds
                contiguous blocks of trials (abc_decoding_qc.cv_decode); label at
                frame t decoded from activity at t - lag, lags over +-90 s.
                Same for the other channel (F_chan2) as a control.
                --regress background|pop removes a global signal (mean Fneu
                over all ROIs / mean of the z-scored traces) from each
                functional trace first (default none; abc_decoding_qc).
    global      the same decoding from the two global signals alone
                (background and population mean): how much of the decoding a
                shared trace explains without single-cell information
    null        lag 0 and best lag repeated with the labels circularly shifted
                (>= 30 s, --n-shuffles)
    splice      median across cells of the F step at frame n - 500 (as found in
                the processed data), and its rank among all frames

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of root>/raw_lag_qc):
summary.png (per target: lag-0 accuracy per session vs label-shift null and
global-signals-only control, as in abc_decoding_qc, and the lag-0 confusion
matrix of each session, goal nodes labelled A/B/C; summary.csv has the
fraction of location frames and of predictions on the goal nodes), lag_scan.png (row per target), lag_scan.csv, summary.csv (row per session and
target; one run per --regress; use separate
--out folders to compare). Inputs are opened read-only.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abc_common import labels, new_figure, safe, save, title
from abc_data.raw import find_raw_sessions
import abc_decoding_qc
from abc_decoding_qc import (apply_lag, circular_error, cv_decode, direct_only, evenness, goal_pull, goal_share,
                              location_labels, plot_accuracy, plot_confusion, segment_route_stats, shifted)
from abc_raw_tuning_qc import regress_out
from suite2p_io import GRIDLINE, INK_MUTED, INK_PRIMARY, SERIES_1, SERIES_2, SERIES_3, default_out_dir, style_axis
import argparse


def zscore(x):
    return np.nan_to_num((x - x.mean(axis=0)) / x.std(axis=0))


def global_signals(s):
    """(T, 2) background (mean Fneu over all ROIs) and mean of the z-scored traces."""
    return np.c_[s.global_signal("background"), zscore(s.activity()).mean(axis=1)]


def features(x, smooth, g=None):
    """z-scored, g (T,) regressed out if given, smoothed."""
    x = zscore(x)
    if g is not None:
        x = zscore(regress_out(x, g)[0])
    if smooth > 1:
        k = np.ones(smooth) / smooth
        x = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, x)
    return x


def trial_folds(s, n_folds):
    tr = s.trial_index()
    n = tr.max() + 1
    edges = np.linspace(0, n, n_folds + 1).round().astype(int)
    fold = np.full(len(tr), -1)
    for f in range(n_folds):
        fold[(tr >= edges[f]) & (tr < edges[f + 1])] = f
    return fold


def step_scores(F, w=60):
    """Median over cells of |mean after - mean before| / pooled SD, per frame."""
    n = F.shape[1]
    c = np.cumsum(np.c_[np.zeros(len(F)), F], 1)
    c2 = np.cumsum(np.c_[np.zeros(len(F)), F ** 2], 1)
    out = np.full(n, np.nan)
    for k in range(w, n - w):
        ma, mb = (c[:, k] - c[:, k - w]) / w, (c[:, k + w] - c[:, k]) / w
        va = (c2[:, k] - c2[:, k - w]) / w - ma ** 2
        vb = (c2[:, k + w] - c2[:, k]) / w - mb ** 2
        out[k] = np.median(np.abs(mb - ma) / np.sqrt((va + vb) / 2 + 1e-9))
    return out


def location_target(s, tracking_dir, exclude_goals):
    """(n_frames,) maze node 1-8 per suite2p frame, -1 for edges, node 9,
    untracked frames (and the goal nodes with exclude_goals); None if the
    session has no tracking output."""
    path = Path(tracking_dir) / f"{safe(s.label)}_imaging_location.npy"
    if not path.exists():
        return None
    loc = np.full(s.n_frames, -1)
    lt = np.load(path)[:s.n_frames]
    loc[:len(lt)] = location_labels(lt, s.task, exclude_goals)
    return loc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="raw ABC folder with funcimg/<session>/preprocessed/suite2p, behaviour/")
    p.add_argument("--out", help="output folder (default ~/Desktop/maze_mini_qc/<name of root>/raw_lag_qc)")
    p.add_argument("--tracking", help="track_position output folder (default "
                                      "~/Desktop/maze_mini_qc/<name of root>/tracking)")
    p.add_argument("--gp-bins", type=int, default=10)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--lag-range", type=float, default=90, help="+-s (default 90)")
    p.add_argument("--lag-step", type=int, default=10, help="frames (default 10)")
    p.add_argument("--n-shuffles", type=int, default=50)
    p.add_argument("--regress", choices=("none", "background", "pop"), default="none",
                   help="global signal regressed out of each functional trace (default none)")
    p.add_argument("--exclude-goals", action="store_true",
                   help="location: leave out frames at the session's A/B/C nodes (5 classes)")
    p.add_argument("--balance-train", nargs="?", const="frames", default="none",
                   choices=("none", "frames", "visits"),
                   help="balance training classes by frames (default when given) or visits")
    p.add_argument("--visit-frames", type=int, default=5,
                   help="--balance-train visits: max frames per visit (default 5)")
    p.add_argument("--average-bins", action="store_true",
                   help="goal progress: decode segment x bin means instead of frames")
    p.add_argument("--max-excess-steps", type=int,
                   help="goal progress: only segments at most N node steps longer than the shortest path "
                        "(routes from the tracking)")
    p.add_argument("--max-wrong-pokes", type=int,
                   help="goal progress: only segments with at most N pokes at non-goal ports")
    a = p.parse_args()
    bal = a.balance_train
    abc_decoding_qc.VISIT_FRAMES = a.visit_frames
    root = os.path.abspath(os.path.expanduser(a.root))
    out = Path(os.path.expanduser(a.out or default_out_dir(root, "raw_lag_qc")))
    out.mkdir(parents=True, exist_ok=True)
    tracking = os.path.expanduser(a.tracking or default_out_dir(root, "tracking"))
    sessions = find_raw_sessions(root)
    if not sessions:
        sys.exit(f"no suite2p sessions below {root}/funcimg")
    print(f"{len(sessions)} session(s) in {root}")

    rows, scans, within, within_glob = [], [], [], []
    rng = np.random.default_rng(0)
    for s in sessions:
        sy = s.imaging_sync
        n = int(round(a.lag_range * s.fs_measured / a.lag_step)) * a.lag_step
        lags = np.arange(-n, n + 1, a.lag_step)
        i0 = int(np.flatnonzero(lags == 0)[0])
        fold = trial_folds(s, a.folds)
        gs = global_signals(s)
        g = None if a.regress == "none" else gs[:, ("background", "pop").index(a.regress)]
        feats = {"functional": features(s.activity(), a.smooth, g),
                 "other channel": features(s.activity(chan2=True), a.smooth),
                 "global signals only": features(gs, a.smooth)}
        gp = s.goal_progress(a.gp_bins)
        if a.max_excess_steps is not None or a.max_wrong_pokes is not None:
            path = Path(tracking) / f"{safe(s.label)}_imaging_location.npy"
            lt = np.full(s.n_frames, np.nan)
            if path.exists():
                v = np.load(path)[:s.n_frames]
                lt[:len(v)] = v
            elif a.max_excess_steps is not None:
                sys.exit(f"--max-excess-steps needs tracking: {path} not found")
            gp = direct_only(gp, segment_route_stats(s, lt), a.max_excess_steps, a.max_wrong_pokes)
        targets = {"goal_progress": gp}
        loc = location_target(s, tracking, a.exclude_goals)
        if loc is None:
            print(f"{s.label}: no tracking in {tracking}, location skipped (run preprocessing/track_position.py)")
        else:
            targets["location"] = loc
        F = np.asarray(s._npy("F"), dtype=np.float64)[s.cells]
        st = step_scores(F)
        k = s.n_frames - 500
        sloc = k - 15 + int(np.nanargmax(st[k - 15:k + 16]))
        base = {"session": s.label, "tag": s.tag, "n_frames": s.n_frames, "daq_edges": sy.n_edges,
                "frames_extrapolated": sy.n_extrapolated, "pulses_daq": len(sy.pulse_times),
                "pulses_pyc": len(s.pycontrol.rsync_times()), "sync_resid_ms": 1e3 * s.pyc_to_daq.max_abs_residual,
                "n_cells": int(s.cells.sum()), "n_trials": s.n_trials, "regress": a.regress,
                "step_n500": float(st[sloc]), "step_n500_rank": float(np.nanmean(st > st[sloc])),
                "step_max_frame": int(np.nanargmax(st))}
        print(f"{s.label}: frames {s.n_frames} vs DAQ edges {sy.n_edges}, pulses {base['pulses_daq']}/"
              f"{base['pulses_pyc']}, resid {base['sync_resid_ms']:.2f} ms, {base['n_cells']} cells, "
              f"{base['n_trials']} trials | step at n-500 {base['step_n500']:.2f} "
              f"(rank {base['step_n500_rank']:.3f})", flush=True)
        min_shift = int(30 * s.fs_measured)
        within.append({})
        within_glob.append({})
        for tgt, y in targets.items():
            avg = a.average_bins and tgt == "goal_progress"
            sc = {name: np.array([cv_decode(*apply_lag(x, y, L), fold, bal, avg)[0] for L in lags])
                  for name, x in feats.items()}
            best = int(lags[np.nanargmax(sc["functional"])])
            null, null_circ = {}, []
            for L in (0, best):
                xl, yl = apply_lag(feats["functional"], y, L)
                res = [cv_decode(xl, shifted(yl, rng, min_shift), fold, bal, avg) for _ in range(a.n_shuffles)]
                null[L] = np.array([r[0] for r in res])
                if L == 0:
                    null_circ = np.array([circular_error(r[1], r[2], a.gp_bins) for r in res])
            row = {**base, "target": tgt, "chance": 1 / len(np.unique(y[y >= 0])),
                   "acc_lag0": sc["functional"][i0], "null_lag0_mean": float(np.nanmean(null[0])),
                   "null_lag0_p95": float(np.nanpercentile(null[0], 95)),
                   "best_lag_frames": best, "acc_best": float(np.nanmax(sc["functional"])),
                   "null_best_p95": float(np.nanpercentile(null[best], 95)),
                   "acc_lag500": float(sc["functional"][np.argmin(np.abs(lags - 500))]),
                   "global_acc_lag0": sc["global signals only"][i0],
                   "global_best_lag": int(lags[np.nanargmax(sc["global signals only"])]),
                   "chan2_acc_lag0": sc["other channel"][i0],
                   "chan2_best_lag": int(lags[np.nanargmax(sc["other channel"])])}
            rows.append(row)
            scans.append((s, tgt, lags, sc, row))
            acc, yt, yp = cv_decode(feats["functional"], y, fold, bal, avg)
            classes = np.unique(y[y >= 0])
            within[-1][tgt] = {"acc": acc, "null": null[0], "y": yt, "pred": yp,
                               "classes": classes, "chance": row["chance"]}
            within_glob[-1][tgt] = {"acc": sc["global signals only"][i0]}
            share = goal_share(yt, yp, s.task) if tgt == "location" else (np.nan, np.nan)
            ev = evenness(yt, yp, classes)
            if tgt == "goal_progress":
                row.update(circ_err_lag0=circular_error(yt, yp, a.gp_bins),
                           null_circ_mean=float(np.mean(null_circ)), null_circ_p5=float(np.percentile(null_circ, 5)))
                print(f"    goal_progress lag 0: circular error {row['circ_err_lag0']:.3f} of a cycle (null mean "
                      f"{row['null_circ_mean']:.3f}, p5 {row['null_circ_p5']:.3f})", flush=True)
            row.update(exclude_goals=a.exclude_goals, balance_train=bal, average_bins=a.average_bins,
                       max_excess_steps=a.max_excess_steps, max_wrong_pokes=a.max_wrong_pokes,
                       task="-".join(map(str, s.task)),
                       frac_true_on_goals=share[0], frac_pred_on_goals=share[1],
                       recall_sd=ev[0], pred_evenness=ev[1],
                       goal_pull=goal_pull(yt, yp, s.task) if tgt == "location" else np.nan)
            if tgt == "location":
                print(f"    location lag 0: goal pull {row['goal_pull']:.2f} (even 0.43)", flush=True)
            print(f"    {tgt} lag 0: recall SD {ev[0]:.3f}, prediction evenness {ev[1]:.2f}", flush=True)
            if tgt == "location" and not a.exclude_goals:
                print(f"    task {row['task']}: location frames on goals {share[0]:.2f}, "
                      f"predictions on goals {share[1]:.2f}", flush=True)
            print(f"  {tgt}: lag 0 {row['acc_lag0']:.3f} (chance {row['chance']:.3f}, null p95 "
                  f"{row['null_lag0_p95']:.3f}), best lag {best} frames {row['acc_best']:.3f}, lag 500 "
                  f"{row['acc_lag500']:.3f} | global only lag 0 {row['global_acc_lag0']:.3f} | "
                  f"other channel lag 0 {row['chan2_acc_lag0']:.3f}", flush=True)

    tgts = [t for t in ("goal_progress", "location") if any(sc[1] == t for sc in scans)]
    note = (f"  |  {a.regress} regressed" if a.regress != "none" else "") + \
        ("  |  goal nodes excluded from location" if a.exclude_goals else "") + \
        (f"  |  training balanced by {bal}" if bal != "none" else "") + \
        ("  |  goal progress on segment x bin means" if a.average_bins else "") + \
        (f"  |  goal progress segments <= {a.max_excess_steps} extra steps" if a.max_excess_steps is not None
         else "") + \
        (f"  |  goal progress segments <= {a.max_wrong_pokes} wrong pokes" if a.max_wrong_pokes is not None
         else "")
    keep = [i for i, w in enumerate(within) if all(t in w for t in tgts)]
    names = [sessions[i].label.split("/")[-1] for i in keep]
    fig, axes = new_figure(len(tgts), 1 + len(keep), figsize=(6.5 * (1 + len(keep)), 5.6 * len(tgts)),
                           squeeze=False)
    for row_axes, tgt in zip(axes, tgts):
        for ax in row_axes:
            style_axis(ax)
        plot_accuracy(row_axes[0], names, [within[i] for i in keep], [within_glob[i] for i in keep], tgt)
        for ax, i, name in zip(row_axes[1:], keep, names):
            plot_confusion(fig, ax, within[i][tgt], tgt,
                           f"{name} (task {'-'.join(map(str, sessions[i].task))})", sessions[i].task)
    save(fig, out / "summary.png", f"Decoding at lag 0, own suite2p on raw data  |  {root}{note}")

    fig, axes = new_figure(len(tgts), len(sessions), figsize=(6.5 * len(sessions), 5.2 * len(tgts)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for s, tgt, lags, sc, r in scans:
        ax = axes[tgts.index(tgt), sessions.index(s)]
        ax.axis("on")
        style_axis(ax)
        t = lags / s.fs_measured
        ax.plot(t, sc["functional"], color=SERIES_1, linewidth=1.6, label="functional channel")
        ax.plot(t, sc["other channel"], color=SERIES_2, linewidth=1.2, label="other channel (control)")
        ax.plot(t, sc["global signals only"], color=SERIES_3, linewidth=1.2, label="global signals only (control)")
        ax.axhline(r["chance"], color=INK_PRIMARY, linestyle="--", linewidth=1, label="chance")
        ax.axhline(r["null_lag0_p95"], color=INK_MUTED, linestyle=":", linewidth=1, label="label-shift null p95")
        ax.axvline(0, color=INK_MUTED, linewidth=1)
        ax.axvline(500 / s.fs_measured, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
        title(ax, f"{tgt.replace('_', ' ')}, {s.label}: {r['n_cells']} cells, {r['n_trials']} trials")
        labels(ax, "lag (s): label at t from activity at t − lag (dashed: 500 frames)", "balanced accuracy")
    save(fig, out / "lag_scan.png", f"Decoding vs lag, own suite2p on raw data  |  {root}{note}")
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.concat([pd.DataFrame({"session": s.label, "target": tgt, "lag_frames": lags, **sc})
               for s, tgt, lags, sc, _ in scans]).to_csv(out / "lag_scan.csv", index=False)
    print(f"Saved figure and tables to {out}")


if __name__ == "__main__":
    main()
