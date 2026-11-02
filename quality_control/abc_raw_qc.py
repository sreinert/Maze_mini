"""
QC of raw ABC sessions processed by us (suite2p on the raw tif; abc_data/raw.py),
everything that works without video tracking. Companion to abc_raw_lag_qc (lag
scan); the suite2p-level checks run with the generic scripts, which take the
same root:

    uv run python quality_control/abc_raw_qc.py /Volumes/behrens/Sandra/mini2p/Arya/RAW_GOOD
    uv run python quality_control/registration_qc.py /Volumes/behrens/Sandra/mini2p/Arya/RAW_GOOD
    uv run python quality_control/suite2p_signal_qc.py /Volumes/behrens/Sandra/mini2p/Arya/RAW_GOOD
    uv run python quality_control/abc_sync_traces.py /Volumes/behrens/Sandra/mini2p/Arya/RAW_GOOD

Checks here (outputs in --out, default ~/Desktop/maze_mini_qc/<name of root>/raw_qc):

    inventory.csv     suite2p frames vs DAQ FrameClock edges, frame intervals,
                      rsync counts DAQ / pyControl / camera, clock-fit residuals,
                      video frames vs GPIO rows, dropped camera frames, ROIs/cells,
                      trials, task; 'flags' lists anything that disagrees
    behaviour.png     pyControl only: phase durations (A->B, B->C, C->A),
                      pokes at non-goal ports per phase, trial rate, goal pokes at
                      the mapped port (abc_behaviour_qc.port_mapping_check)
    peth/<s>.png      PETHs of all ROIs (F - 0.7 Fneu, z-scored) at A/B/C_on,
                      even trials sorted by odd (cross-validated), and odd/even
                      reliability vs a per-cell circular-shift null
                      (abc_neural_task_qc); peth.csv per cell
    video_check.png   camera frames at an A, B and C poke per session (through the
                      rsync fit), labelled with the goal port: the mouse should be
                      at that port. No tracking needed.

Needs tracking, so not here: spatial maps, location decoding, route excess,
speed coupling.
"""
import numpy as np
import pandas as pd

from abc_behaviour_qc import port_mapping_check
from abc_common import (GOAL_COLORS, GOAL_NAMES, PHASE_NAMES, labels, new_figure, parser, safe, save,
                        setup, title)
from abc_data import maze
from abc_data.pycontrol import trial_events
from abc_inventory import _video_frames
from abc_neural_task_qc import POST_S, PRE_S, _cv_heatmap, _rel_hist, peth, peth_reliability
from suite2p_io import GRIDLINE, INK_PRIMARY, SERIES_1, style_axis


def inventory_row(s):
    flags = []
    sy, log = s.imaging_sync, s.pycontrol
    ifi = np.diff(sy.frame_times[:sy.n_edges])
    r = {"session": s.label, "tag": s.tag, "task_name": log.info.get("task_name"),
         "task_ABC": "-".join(map(str, s.task)), "pyc_duration_s": round(log.duration, 1),
         "n_trials": s.n_trials, "s2p_frames": s.n_frames, "h5_frame_edges": sy.n_edges,
         "frames_extrapolated": sy.n_extrapolated, "fs_ops": s.ops.get("fs"), "fs_measured": round(s.fs_measured, 4),
         "ifi_min_ms": round(1e3 * ifi.min(), 2), "ifi_max_ms": round(1e3 * ifi.max(), 2),
         "pulses_h5": len(sy.pulse_times), "pulses_pyc": len(log.rsync_times()),
         "sync_resid_daq_ms": round(1e3 * s.pyc_to_daq.max_abs_residual, 2),
         "n_roi": len(s.cells), "n_cells": int(s.cells.sum()), "n_iscell": int(s.iscell.sum())}
    if s.n_frames - sy.n_edges not in (0, 1, 2):
        flags.append("suite2p frames != h5 frame edges")
    if r["pulses_h5"] != r["pulses_pyc"]:
        flags.append("rsync counts differ DAQ/pyControl")
    if s.gpio:
        cam = s.camera_sync
        cifi = cam.frame_intervals
        n_vid, fps, w, h = _video_frames(s.video_path, expected=len(cam.frame_times))
        r.update({"pulses_camera": len(cam.pulse_times), "video_frames": n_vid, "gpio_rows": len(cam.frame_times),
                  "video_extra_frames": n_vid - len(cam.frame_times), "video_fps": fps,
                  "camera_dropped": int(np.sum(cifi > 1.5 * np.median(cifi))),
                  "sync_resid_camera_ms": round(1e3 * s.pyc_to_camera.max_abs_residual, 2)})
        if r["pulses_camera"] != r["pulses_pyc"]:
            flags.append("rsync counts differ camera/pyControl")
        if r["camera_dropped"]:
            flags.append(f"{r['camera_dropped']} dropped camera frames")
        if r["video_extra_frames"]:
            flags.append(f"video has {r['video_extra_frames']:+d} frames vs GPIO rows")
    else:
        flags.append("no video/GPIO")
    r["flags"] = "; ".join(flags)
    return r


def behaviour_table(s):
    """Per trial and phase (pyControl only): duration and pokes at non-goal ports."""
    ev = trial_events(s.pycontrol)
    pokes = s.pycontrol.pokes()
    nodes = np.array([maze.port_to_node(p) for p in pokes.port])
    goals = list(s.task)
    rows = []
    for i, tr in enumerate(ev):
        for ph in range(3):
            t0, t1 = tr[ph], tr[ph + 1]
            if not (np.isfinite(t0) and np.isfinite(t1)):
                continue
            m = (pokes.time > t0) & (pokes.time < t1)
            wrong = int(np.sum(m & ~np.isin(nodes, [goals[ph], goals[(ph + 1) % 3]])))
            rows.append({"session": s.label, "trial": i, "phase": PHASE_NAMES[ph], "duration_s": t1 - t0,
                         "wrong_pokes": wrong})
    return pd.DataFrame(rows)


def peth_analysis(s, n_shuffles, seed=0):
    x = s.activity()
    x = np.nan_to_num((x - x.mean(0)) / x.std(0))
    pre, post = int(round(PRE_S * s.fs_measured)), int(round(POST_S * s.fs_measured))
    events = s.goal_frames()
    rel = peth_reliability(x, events, pre, post)
    rng = np.random.default_rng(seed)
    lo = int(30 * s.fs_measured)
    null = np.zeros((n_shuffles,) + rel.shape)
    for k in range(n_shuffles):
        sh = rng.integers(lo, len(x) - lo, x.shape[1])
        null[k] = peth_reliability(np.stack([np.roll(x[:, i], sh[i]) for i in range(x.shape[1])], 1),
                                   events, pre, post)
    return {"x": x, "peths": [peth(x, ev, pre, post) for ev in events], "rel": rel,
            "null": null, "thr": np.nanpercentile(null, 95, axis=0)}


def plot_peth(s, r, out_path):
    fig, axes = new_figure(2, 3, figsize=(17, 10))
    ext = [-PRE_S, POST_S, r["x"].shape[1], 0]
    im = None
    for j, (name, P) in enumerate(zip(GOAL_NAMES, r["peths"])):
        ax = axes[0, j]
        style_axis(ax)
        if len(P) >= 4:
            im = _cv_heatmap(ax, P[0::2].mean(0), P[1::2].mean(0), ext, vlim=0.8)
        ax.axvline(0, color=INK_PRIMARY, linewidth=1)
        title(ax, f"{name}_on (node {s.task[j]}): even trials, sorted by odd ({len(P)} events)")
        labels(ax, "time from poke (s)", "cell" if j == 0 else None)
        style_axis(axes[1, j])
        _rel_hist(axes[1, j], r["rel"][j], r["null"][:, j].ravel(), r["thr"][j], f"{name}_on PETH reliability")
    if im is not None:
        fig.colorbar(im, ax=axes[0, 2], label="mean z-scored activity")
    save(fig, out_path, f"{s.label}  |  {r['x'].shape[1]} cells, own suite2p, task {'-'.join(map(str, s.task))}")


def plot_behaviour(sessions, beh, per, out_path, root):
    names = [s.label.split("/")[-1] for s in sessions]
    x = np.arange(len(names))
    fig, axes = new_figure(1, 4, figsize=(22, 5.5))
    for ax in axes:
        style_axis(ax)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    for p, (ph, c) in enumerate(zip(PHASE_NAMES, GOAL_COLORS)):
        d = [beh[(beh.session == s.label) & (beh.phase == ph)] for s in sessions]
        for ax, col, f in ((axes[0], "duration_s", np.median), (axes[1], "wrong_pokes", np.mean)):
            ax.plot(x + 0.12 * (p - 1), [f(v[col]) if len(v) else np.nan for v in d], "o-", color=c, markersize=7,
                    linewidth=1.2, markeredgecolor="white", markeredgewidth=1.5, label=ph, zorder=3)
    axes[0].set_yscale("log")
    title(axes[0], "Phase duration")
    labels(axes[0], y="median duration (s)")
    title(axes[1], "Pokes at non-goal ports")
    labels(axes[1], y="pokes per phase")
    axes[2].plot(x, per.trials_per_min, "o-", color=SERIES_1, markersize=7, markeredgecolor="white",
                 markeredgewidth=1.5, zorder=3)
    title(axes[2], "Trial rate")
    labels(axes[2], y="trials per minute")
    axes[3].plot(x, per.port_mapping_ok, "o", color=SERIES_1, markersize=9, markeredgecolor="white",
                 markeredgewidth=1.5, zorder=3)
    axes[3].set_ylim(0, 1.05)
    title(axes[3], "Goal pokes at the mapped port")
    labels(axes[3], y="fraction of A/B/C_on")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    for ax in axes[:2]:
        ax.legend(frameon=False, fontsize=8)
    save(fig, out_path, f"Behaviour from pyControl  |  {root}")


def video_check(sessions, out_path, width=400):
    import cv2
    h = int(width * 1024 / 1280)
    grid = []
    for s in sessions:
        tiles = []
        cap = cv2.VideoCapture(s.video_path) if s.video_path else None
        for name, node in zip(GOAL_NAMES, s.task):
            tile = np.full((h, width, 3), 40, np.uint8)
            times = s.pycontrol.state_times(f"{name}_on")
            if cap is not None and len(times):
                t = times[len(times) // 2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(s.pyc_to_camera_frame(t)))
                ok, fr = cap.read()
                if ok:
                    tile = cv2.resize(fr, (width, h))
            cv2.putText(tile, f"{s.label.split('/')[-1]} {name}_on at port {maze.node_to_port(node)} (node {node})",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        if cap is not None:
            cap.release()
        grid.append(np.hstack(tiles))
    img = np.vstack(grid)
    head = np.full((34, img.shape[1], 3), 252, np.uint8)
    cv2.putText(head, "Video at goal pokes (no tracking): the mouse should be at the named port "
                      "(A1 B1 C1 = bottom row, left to right)", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (11, 11, 11), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([head, img]))


def main():
    p = parser(__doc__, "raw_qc")
    p.add_argument("--n-shuffles", type=int, default=200)
    a = p.parse_args()
    root, out, sessions = setup(a, "raw_qc")
    (out / "peth").mkdir(exist_ok=True)
    inv, beh, per, cells = [], [], [], []
    for s in sessions:
        r = inventory_row(s)
        inv.append(r)
        b = behaviour_table(s)
        beh.append(b)
        per.append({"session": s.label, "n_trials": s.n_trials,
                    "trials_per_min": 60 * s.n_trials / s.pycontrol.duration,
                    "port_mapping_ok": port_mapping_check(s),
                    "median_phase_s": b.duration_s.median(), "wrong_pokes_per_phase": b.wrong_pokes.mean()})
        pr = peth_analysis(s, a.n_shuffles)
        plot_peth(s, pr, out / "peth" / f"{safe(s.label)}.png")
        frac = [float(np.mean(pr["rel"][j][np.isfinite(pr["rel"][j])] > pr["thr"][j][np.isfinite(pr["rel"][j])]))
                for j in range(3)]
        per[-1].update({f"frac_reliable_{n}": f for n, f in zip(GOAL_NAMES, frac)})
        cells.append(pd.DataFrame({"session": s.label, "cell": np.flatnonzero(s.cells),
                                   **{f"rel_{n}": pr["rel"][j] for j, n in enumerate(GOAL_NAMES)},
                                   **{f"null95_{n}": pr["thr"][j] for j, n in enumerate(GOAL_NAMES)}}))
        print(f"{s.label}: flags: {r['flags'] or 'none'} | {s.n_trials} trials, {per[-1]['trials_per_min']:.1f}/min, "
              f"port mapping {per[-1]['port_mapping_ok']:.0%} | reliable PETH A/B/C "
              f"{frac[0]:.0%}/{frac[1]:.0%}/{frac[2]:.0%} (chance 5 %)", flush=True)
    inv = pd.DataFrame(inv)
    inv.to_csv(out / "inventory.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(inv.set_index("session").T.to_string())
    beh = pd.concat(beh, ignore_index=True)
    beh.to_csv(out / "trials.csv", index=False)
    per = pd.DataFrame(per)
    per.to_csv(out / "behaviour.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(out / "peth.csv", index=False)
    plot_behaviour(sessions, beh, per, out / "behaviour.png", root)
    video_check(sessions, out / "video_check.png")
    print(f"Saved figures and tables to {out}")


if __name__ == "__main__":
    main()
