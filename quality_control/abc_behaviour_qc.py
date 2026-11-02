"""
Behaviour checks for the ABC-task data: tracking quality, consistency of the
location labels, task performance across sessions, and a look at the video at
the poke times.

    uv run python quality_control/abc_behaviour_qc.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of root>/behaviour_qc):

    sessions/<session>.png  per session:
        trajectory       tracked XY (aligned frames) coloured by task phase, goal
                         nodes marked
        occupancy        fraction of time at each location id (-1 = untracked)
        speed            head speed over the session, tracking gaps marked
        label check      distance from XY to the centroid of its labelled
                         location vs to the nearest other centroid: labels that
                         disagree with the position show up right of the diagonal
        phase durations  per trial, A->B / B->C / C->A
        excess steps     nodes visited per phase minus the shortest path
                         (0 = optimal route; edges 11/14 do not exist, maze.py)
    summary.png             across sessions: phase durations, optimal-route
                            fraction, wrong-port pokes, trial rate, tracking loss,
                            label consistency, port mapping check
    video_check.png         camera frames at A/B/C_on (one per goal per session,
                            mapped through the rsync clock fit) with the tracked
                            position of that aligned frame (ring) and the goal
                            node (square): the mouse should be at the goal port.
                            The last column is a B_on with tracking lost, if any.
    behaviour.csv           per-session numbers
    trials.csv              per-trial phase durations, steps, wrong pokes

Port mapping check: every A/B/C_on must be triggered by a poke at the port that
maze.node_to_port gives for the goal node (fraction reported, should be 1).
Inputs are opened read-only.
"""
import numpy as np
import pandas as pd

from abc_common import (GOAL_COLORS, GOAL_NAMES, PHASE_NAMES, labels, new_figure, parser, safe, save,
                        setup, title)
from abc_data import maze
from abc_data.sync import time_to_frame
from suite2p_io import GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SERIES_1, SERIES_2, style_axis

JUMP_PX = 150  # head displacement between two imaging frames (133 ms) counted as a tracking jump


def trial_table(s):
    """Per trial and phase: duration, nodes on the route, shortest path, wrong pokes."""
    D = maze.node_distances()
    tt = s.trial_times
    pokes = s.pycontrol.pokes()
    poke_frames = s.pyc_to_aligned_frame(pokes.time.to_numpy())
    poke_nodes = np.array([maze.port_to_node(p) for p in pokes.port])
    goals = list(s.task)
    t = s.time
    rows = []
    for i, tr in enumerate(tt):
        for ph in range(3):
            start, end = tr[ph], tr[ph + 1]
            src, dst = goals[ph], goals[(ph + 1) % 3]
            complete = end < s.n_frames or ph < 2
            path = maze.node_path(s.location[start:min(end, s.n_frames - 1) + 1])
            in_phase = (poke_frames > start) & (poke_frames < end)
            wrong = int(np.sum(in_phase & ~np.isin(poke_nodes, [src, dst])))
            rows.append({"session": s.label, "trial": i, "phase": PHASE_NAMES[ph],
                         "complete": bool(complete and not (i == 0 and ph == 0)),
                         "duration_s": float(t[min(end, s.n_frames - 1)] - t[start]),
                         "steps": max(len(path) - 1, 0), "shortest": float(D[src, dst]),
                         "wrong_pokes": wrong})
    df = pd.DataFrame(rows)
    df["excess_steps"] = df.steps - df.shortest
    return df


def port_mapping_check(s):
    """Fraction of A/B/C_on events whose last preceding poke is at the goal port."""
    pokes = s.pycontrol.pokes()
    ok, n = 0, 0
    for name, node in zip(GOAL_NAMES, s.task):
        for t in s.pycontrol.state_times(f"{name}_on"):
            prev = pokes[pokes.time <= t]
            n += 1
            ok += int(len(prev) > 0 and prev.port.iloc[-1] == maze.node_to_port(node))
    return ok / n if n else np.nan


def tracking_stats(s):
    xy = np.asarray(s.xy, dtype=float)
    good = np.all(np.isfinite(xy), axis=1)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cent = maze.location_centroids(xy, s.location)
    ids = np.array(list(cent))
    C = np.array([cent[i] for i in ids])
    m = good & (s.location != maze.UNKNOWN) & np.isin(s.location, ids)
    d_all = np.linalg.norm(xy[m][:, None, :] - C[None], axis=2)
    lab_idx = np.searchsorted(ids, s.location[m]) if np.all(np.diff(ids) > 0) else None
    d_lab = d_all[np.arange(m.sum()), lab_idx]
    d_other = np.where(np.arange(len(ids))[None] == lab_idx[:, None], np.inf, d_all).min(axis=1)
    return {"xy_nan_frac": float(1 - good.mean()),
            "loc_unknown_frac": float(np.mean(s.location == maze.UNKNOWN)),
            "loc_without_xy": int(np.sum(~good & (s.location != maze.UNKNOWN))),
            "jumps": int(np.nansum(step > JUMP_PX)),
            "label_consistent_frac": float(np.mean(d_lab <= d_other)),
            "_d_lab": d_lab, "_d_other": d_other, "_centroids": cent,
            "_speed": np.r_[np.nan, step] * s.fs_measured}


def plot_session(s, trials, ts, out_path):
    fig, axes = new_figure(2, 3, figsize=(19, 11))
    a = axes.ravel()
    for ax in a:
        style_axis(ax)
    xy = np.asarray(s.xy, dtype=float)
    ph = s.phase()
    cent = ts["_centroids"]

    ax = a[0]
    for p, (name, c) in enumerate(zip(PHASE_NAMES, GOAL_COLORS)):
        seg = np.where(ph == p, 1.0, np.nan)[:, None] * xy
        ax.plot(seg[:, 0], seg[:, 1], color=c, linewidth=0.6, alpha=0.6, label=name)
    for l, (x, y) in cent.items():
        ax.text(x, y, maze.location_name(l), fontsize=7, color=INK_SECONDARY, ha="center", va="center")
    for name, node, c in zip(GOAL_NAMES, s.task, GOAL_COLORS):
        if node in cent:
            ax.plot(*cent[node], "s", markersize=16, markerfacecolor="none", markeredgecolor=c, markeredgewidth=2.5)
            ax.annotate(name, cent[node], xytext=(14, 14), textcoords="offset points", color=INK_PRIMARY,
                        fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1280)
    ax.set_ylim(1024, 0)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    title(ax, f"Trajectory by phase (task {'-'.join(map(str, s.task))})")
    labels(ax, "x (video px)", "y (video px)")

    ax = a[1]
    ids = [maze.UNKNOWN] + list(maze.LOCATIONS)
    occ = [np.mean(s.location == l) for l in ids]
    ax.bar(range(len(ids)), occ, color=[INK_MUTED] + [SERIES_1] * 9 + [SERIES_2] * 12, edgecolor="white")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(["-1"] + [maze.location_name(l) for l in maze.LOCATIONS], rotation=90, fontsize=7.5)
    ax.plot([], [], "s", color=SERIES_1, label="nodes")
    ax.plot([], [], "s", color=SERIES_2, label="edges")
    ax.plot([], [], "s", color=INK_MUTED, label="untracked")
    ax.legend(frameon=False, fontsize=8)
    title(ax, "Occupancy")
    labels(ax, y="fraction of frames")

    ax = a[2]
    sp = ts["_speed"]
    ax.plot(s.time, sp, color=SERIES_1, linewidth=0.6)
    gaps = ~np.all(np.isfinite(xy), axis=1)
    ymax = np.nanpercentile(sp, 99.9)
    ax.plot(s.time[gaps], np.full(gaps.sum(), ymax * 1.05), "|", color=SERIES_2, markersize=6,
            label=f"no tracking ({gaps.mean():.1%})")
    ax.axhline(JUMP_PX * s.fs_measured, color=INK_MUTED, linestyle="--", linewidth=1,
               label=f"jump threshold ({ts['jumps']} jumps)")
    ax.set_ylim(0, max(ymax * 1.1, JUMP_PX * s.fs_measured * 1.1))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    title(ax, "Head speed")
    labels(ax, "time (s)", "speed (px/s)")

    ax = a[3]
    dl, do = ts["_d_lab"], ts["_d_other"]
    ax.scatter(do, dl, s=3, color=SERIES_1, alpha=0.25, edgecolor="none")
    lim = max(np.percentile(dl, 99.9), np.percentile(do, 99.9))
    ax.plot([0, lim], [0, lim], color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    title(ax, f"Label check: {ts['label_consistent_frac']:.1%} of frames nearest their label")
    labels(ax, "distance to nearest other location centroid (px)", "distance to labelled centroid (px)")

    for ax, col, ylab in ((a[4], "duration_s", "duration (s)"), (a[5], "excess_steps", "nodes beyond shortest path")):
        for p, (name, c) in enumerate(zip(PHASE_NAMES, GOAL_COLORS)):
            d = trials[(trials.phase == name) & trials.complete]
            ax.plot(d.trial, d[col], "o-", color=c, markersize=5, linewidth=1.2, label=name,
                    markeredgecolor="white", markeredgewidth=1)
        if col == "duration_s":
            ax.set_yscale("log")
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
        labels(ax, "trial", ylab)
    title(a[4], "Phase durations")
    title(a[5], "Route excess")
    save(fig, out_path, f"{s.label}  |  {len(s.trial_times)} trials in {s.time[-1]:.0f} s")


def video_check(sessions, out_path, width=400):
    import cv2
    cols = ["A_on", "B_on", "C_on", "B_on, no tracking"]
    h = int(width * 1024 / 1280)
    grid = []
    for s in sessions:
        cap = cv2.VideoCapture(s.video_path)
        xy = np.asarray(s.xy, dtype=float)
        cent = maze.location_centroids(xy, s.location)
        tiles = []
        for j, col in enumerate(cols):
            name = col.split(",")[0]
            node = s.task[GOAL_NAMES.index(name[0])]
            times = s.pycontrol.state_times(name)
            k_all = s.pyc_to_aligned_frame(times).astype(int)
            valid = (k_all >= 0) & (k_all < s.n_frames)
            if j == 3:
                valid &= ~np.all(np.isfinite(xy[np.clip(k_all, 0, s.n_frames - 1)]), axis=1)
            idx = np.flatnonzero(valid)
            tile = np.full((h, width, 3), 40, np.uint8)
            if len(idx):
                i = idx[len(idx) // 2]
                vf = int(time_to_frame(s.pyc_to_camera(times[i]), s.camera_sync.frame_times))
                cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
                ok, fr = cap.read()
                if ok:
                    p = xy[k_all[i]]
                    if node in cent:
                        c = cent[node].astype(int)
                        cv2.rectangle(fr, tuple(c - 40), tuple(c + 40), (52, 104, 235), 4)  # orange (BGR)
                    if np.all(np.isfinite(p)):
                        cv2.circle(fr, (int(p[0]), int(p[1])), 22, (214, 120, 42), 5)   # blue (BGR)
                    tile = cv2.resize(fr, (width, h))
                    cv2.putText(tile, f"{s.label.split('/')[-1]} {col} #{i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        cap.release()
        grid.append(np.hstack(tiles))
    img = np.vstack(grid)
    head = np.full((34, img.shape[1], 3), 252, np.uint8)
    cv2.putText(head, "Video at goal pokes: ring = tracked head (blue), square = goal node (orange)",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (11, 11, 11), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([head, img]))


def plot_summary(per, trials, out_path, root):
    names = [l.split("/")[-1] for l in per.session]
    x = np.arange(len(names))
    fig, axes = new_figure(2, 3, figsize=(19, 10))
    a = axes.ravel()
    for ax in a:
        style_axis(ax)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    done = trials[trials.complete]

    def by_phase(ax, fn, ylab, ttl, log=False):
        for p, (name, c) in enumerate(zip(PHASE_NAMES, GOAL_COLORS)):
            y = [fn(done[(done.session == s) & (done.phase == name)]) for s in per.session]
            ax.plot(x + 0.12 * (p - 1), y, "o-", color=c, markersize=7, linewidth=1.2, label=name,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        if log:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
        title(ax, ttl)
        labels(ax, y=ylab)

    by_phase(a[0], lambda d: d.duration_s.median(), "median duration (s)", "Phase duration", log=True)
    by_phase(a[1], lambda d: np.mean(d.excess_steps == 0), "fraction of phases", "Shortest route taken")
    by_phase(a[2], lambda d: d.wrong_pokes.mean(), "pokes per phase", "Pokes at non-goal ports")

    ax = a[3]
    ax.plot(x, per.trials_per_min, "o-", color=SERIES_1, markersize=7, markeredgecolor="white",
            markeredgewidth=1.5, zorder=3)
    title(ax, "Trial rate")
    labels(ax, y="trials per minute")

    ax = a[4]
    ax.plot(x, per.xy_nan_frac, "o-", color=SERIES_1, markersize=7, label="XY missing",
            markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax.plot(x, 1 - per.label_consistent_frac, "o-", color=SERIES_2, markersize=7,
            label="label not nearest centroid", markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax.legend(frameon=False, fontsize=8)
    title(ax, "Tracking and labels")
    labels(ax, y="fraction of frames")

    ax = a[5]
    ax.plot(x, per.port_mapping_ok, "o", color=SERIES_1, markersize=9, markeredgecolor="white",
            markeredgewidth=1.5, zorder=3)
    ax.set_ylim(0, 1.05)
    title(ax, "Goal pokes at the mapped port (maze.node_to_port)")
    labels(ax, y="fraction of A/B/C_on")
    for ax in a:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    save(fig, out_path, f"Behaviour across sessions  |  {root}")


def main():
    p = parser(__doc__, "behaviour_qc")
    p.add_argument("--no-video", action="store_true", help="skip the video check montage")
    a = p.parse_args()
    root, out, sessions = setup(a, "behaviour_qc")
    (out / "sessions").mkdir(exist_ok=True)
    per, all_trials = [], []
    for s in sessions:
        trials = trial_table(s)
        ts = tracking_stats(s)
        all_trials.append(trials)
        done = trials[trials.complete]
        row = {"session": s.label, "task": "-".join(map(str, s.task)), "n_trials": len(s.trial_times),
               "trials_per_min": 60 * len(s.trial_times) / s.time[-1],
               "port_mapping_ok": port_mapping_check(s),
               "median_phase_s": done.duration_s.median(), "frac_shortest": float(np.mean(done.excess_steps == 0)),
               "wrong_pokes_per_phase": done.wrong_pokes.mean(),
               **{k: v for k, v in ts.items() if not k.startswith("_")}}
        per.append(row)
        plot_session(s, trials, ts, out / "sessions" / f"{safe(s.label)}.png")
        print(f"{s.label}: {row['n_trials']} trials, {row['trials_per_min']:.1f}/min, shortest route "
              f"{row['frac_shortest']:.0%}, XY missing {row['xy_nan_frac']:.1%}, labels consistent "
              f"{row['label_consistent_frac']:.1%}, {row['jumps']} jumps, port mapping {row['port_mapping_ok']:.0%}")
    per = pd.DataFrame(per)
    trials = pd.concat(all_trials, ignore_index=True)
    per.to_csv(out / "behaviour.csv", index=False)
    trials.to_csv(out / "trials.csv", index=False)
    plot_summary(per, trials, out / "summary.png", root)
    if not a.no_video:
        video_check(sessions, out / "video_check.png")
    print(f"Saved figures and tables to {out}")


if __name__ == "__main__":
    main()
