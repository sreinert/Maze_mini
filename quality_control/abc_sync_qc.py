"""
Synchronisation and alignment checks for the ABC-task data: are behaviour,
video and neural activity in the aligned arrays on the same clock?

    uv run python quality_control/abc_sync_qc.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding

Figure sync_summary.png, one panel per check:

    timeline        imaging, video, pyControl run and the aligned window on the
                    pyControl clock: the window should start at pyControl t = 0
                    and lie inside imaging and video
    clock fits      max |residual| of the linear rsync fits pyControl -> imaging
                    DAQ and pyControl -> camera (sub-ms for the DAQ; the camera is
                    limited by its 16.7 ms frame period)
    clock drift     slope - 1 of those fits in ppm (crystal drift; tiny is fine)
    frame rate      imaging frame intervals against the nominal 1/fs: the real
                    rate is ~7.52 Hz, not the nominal 7.5 Hz in extraction_metadata
    trial times     given trial_times minus the same events recomputed from
                    pyControl through the DAQ sync (nearest frame): should be 0
    neural offset   median correlation between each suite2p ROI's dff and its
                    best ROICaT column, for window offsets around the
                    behaviour-derived offset: a sharp peak at 0 means the
                    aligned activity is on the same frames as the behaviour
    goal location   fraction of A/B/C_on events at which maze_location equals the
                    goal node (tracking + alignment check)

sync.csv holds the numbers per session. Inputs are opened read-only.
"""
import numpy as np
import pandas as pd

from abc_common import GOAL_COLORS, GOAL_NAMES, labels, new_figure, parser, save, setup, title
from suite2p_io import (GRIDLINE, INK_MUTED, INK_PRIMARY, SERIES_1, SERIES_2, SERIES_3, SERIES_4,
                        colormap, style_axis)

OFFSETS = np.arange(-5, 6)


def session_sync(s):
    log, cam, sy = s.pycontrol, s.camera_sync, s.imaging_sync
    daq, camfit = s.pyc_to_daq, s.pyc_to_camera
    resync = s.trial_times_resynced()
    given = s.trial_times.astype(float)
    diff = resync - given
    diff[0, 0] = np.nan                  # first A is set to 0 in trial_times
    if np.isnan(resync[-1, 3]):
        diff[-1, 3] = np.nan             # last trial has no next A (ends at T)
    match_r = [float(np.median(s.roi_match(offset=s.aligned_offset + o).best_r)) for o in OFFSETS]
    loc_ok = []
    for j, node in enumerate(s.task):   # columns A, B, C; the first trial's A is 0, not an event
        k = s.trial_times[1:, 0] if j == 0 else s.trial_times[:, j]
        loc_ok.append(float(np.mean(s.location[np.clip(k, 0, s.n_frames - 1)] == node)))
    return {
        "session": s.label,
        "imaging_span": ((sy.frame_times[0] - daq.intercept) / daq.slope,
                         (sy.frame_times[-1] - daq.intercept) / daq.slope),
        "video_span": ((cam.frame_times[0] - camfit.intercept) / camfit.slope,
                       (cam.frame_times[-1] - camfit.intercept) / camfit.slope),
        "pyc_span": (0.0, log.duration),
        "window_span": (float(s.time[0]), float(s.time[-1])),
        "resid_daq_ms": 1e3 * daq.max_abs_residual,
        "resid_camera_ms": 1e3 * camfit.max_abs_residual,
        "drift_daq_ppm": 1e6 * (daq.slope - 1),
        "drift_camera_ppm": 1e6 * (camfit.slope - 1),
        "ifi_ms": 1e3 * np.diff(sy.frame_times[:sy.n_edges]),
        "fs_nominal": s.fs, "fs_measured": s.fs_measured,
        "trial_diff": diff[np.isfinite(diff)],
        "match_r": match_r,
        "loc_ok": loc_ok,
        "aligned_offset": s.aligned_offset,
        "offset_at_pyc_start": s.offset_at_pyc_start,
    }


def _strip(ax, names, rows, keys, legend, colors, ylabel, log=False):
    for i, (key, name, c) in enumerate(zip(keys, legend, colors)):
        # second series hollow and larger, so coinciding values stay visible
        ax.plot(range(len(names)), [r[key] for r in rows], "o", color=c, markersize=7 if i == 0 else 11,
                markerfacecolor=c if i == 0 else "none", markeredgecolor="white" if i == 0 else c,
                markeredgewidth=1.5, label=name, zorder=3 if i == 0 else 4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    labels(ax, y=ylabel)
    if len(keys) > 1:
        ax.legend(frameon=False, fontsize=8)


def plot_summary(rows, out_path, root):
    names = [r["session"].split("/")[-1] for r in rows]
    fig, axes = new_figure(2, 4, figsize=(22, 10))
    a = axes.ravel()
    for ax in a:
        style_axis(ax)

    ax = a[0]
    spans = [("imaging_span", "imaging", SERIES_1), ("video_span", "video", SERIES_2),
             ("pyc_span", "pyControl run", SERIES_3), ("window_span", "aligned window", SERIES_4)]
    for i, r in enumerate(rows):
        for j, (key, name, c) in enumerate(spans):
            lo, hi = r[key]
            ax.plot([lo, hi], [i + 0.18 * (j - 1.5)] * 2, color=c, linewidth=3, solid_capstyle="round",
                    label=name if i == 0 else None)
    ax.axvline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_ylim(len(rows) + 0.6, -0.6)   # room for the legend below the last session
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=4)
    title(ax, "Recording spans on the pyControl clock")
    labels(ax, x="pyControl time (s)")

    _strip(a[1], names, rows, ["resid_daq_ms", "resid_camera_ms"], ["imaging DAQ", "camera"],
           [SERIES_1, SERIES_2], "max |residual| (ms)", log=True)
    title(a[1], "rsync clock fits from pyControl")
    _strip(a[2], names, rows, ["drift_daq_ppm", "drift_camera_ppm"], ["imaging DAQ", "camera"],
           [SERIES_1, SERIES_2], "slope - 1 (ppm)")
    title(a[2], "Clock drift relative to pyControl")

    ax = a[3]
    ifi = np.concatenate([r["ifi_ms"] for r in rows])
    ax.hist(ifi, bins=40, color=SERIES_1, edgecolor="white", linewidth=0.5)
    nominal = 1e3 / rows[0]["fs_nominal"]
    ax.axvline(nominal, color=SERIES_2, linewidth=2, label=f"nominal 1/fs = {nominal:.1f} ms")
    ax.axvline(np.median(ifi), color=INK_PRIMARY, linewidth=1.5, linestyle="--",
               label=f"measured median {np.median(ifi):.2f} ms ({1e3 / np.median(ifi):.3f} Hz)")
    ax.legend(frameon=False, fontsize=8)
    title(ax, "Imaging frame intervals (all sessions)")
    labels(ax, x="interval (ms)", y="frames")

    ax = a[4]
    vals = np.arange(-3, 4)
    w = 0.8 / len(rows)
    cm = colormap()
    for i, r in enumerate(rows):
        cnt = [np.sum(r["trial_diff"] == v) for v in vals]
        ax.bar(vals + (i - len(rows) / 2 + 0.5) * w, cnt, width=w, color=cm(0.25 + 0.75 * i / max(len(rows) - 1, 1)),
               label=names[i])
    ax.set_xticks(vals)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    title(ax, "Given trial_times − resynced (A/B/C_on)")
    labels(ax, x="difference (imaging frames)", y="events")

    ax = a[5]
    M = np.array([r["match_r"] for r in rows])
    im = ax.imshow(M, aspect="auto", cmap=colormap(), vmin=0, vmax=1,
                   extent=[OFFSETS[0] - 0.5, OFFSETS[-1] + 0.5, len(rows) - 0.5, -0.5])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(rows)):
        for j, o in enumerate(OFFSETS):
            ax.text(o, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.6 else INK_PRIMARY)
    fig.colorbar(im, ax=ax, label="median best-match r")
    title(ax, "Aligned activity vs suite2p dff, by frame shift")
    labels(ax, x="shift from behaviour-derived offset (frames)")

    ax = a[6]
    for j, (name, c) in enumerate(zip(GOAL_NAMES, GOAL_COLORS)):
        ax.plot(np.arange(len(rows)) + 0.15 * (j - 1), [r["loc_ok"][j] for r in rows], "o", color=c,
                markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=f"{name}_on")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8)
    title(ax, "maze_location = goal node at the poke frame")
    labels(ax, y="fraction of events")

    ax = a[7]
    _strip(ax, names, rows, ["aligned_offset", "offset_at_pyc_start"],
           ["from trial_times", "nearest frame to pyControl t=0"], [SERIES_1, SERIES_2],
           "suite2p frame of aligned frame 0")
    title(ax, "Aligned window start")
    save(fig, out_path, f"Sync and alignment  |  {root}")


def main():
    a = parser(__doc__, "sync_qc").parse_args()
    root, out, sessions = setup(a, "sync_qc")
    rows = []
    for s in sessions:
        r = session_sync(s)
        rows.append(r)
        print(f"{s.label}: DAQ resid {r['resid_daq_ms']:.2f} ms, camera {r['resid_camera_ms']:.1f} ms, "
              f"trial-time diffs {np.unique(r['trial_diff']).astype(int).tolist()}, "
              f"match r at offset {r['match_r'][5]:.3f} (next best {max(np.delete(r['match_r'], 5)):.3f}), "
              f"goal location {np.round(r['loc_ok'], 2).tolist()}")
    plot_summary(rows, out / "sync_summary.png", root)
    table = pd.DataFrame([{
        "session": r["session"], "aligned_offset": r["aligned_offset"],
        "offset_at_pyc_start": r["offset_at_pyc_start"],
        "window_start_pyc_s": r["window_span"][0], "window_end_pyc_s": r["window_span"][1],
        "imaging_start_pyc_s": r["imaging_span"][0], "imaging_end_pyc_s": r["imaging_span"][1],
        "video_start_pyc_s": r["video_span"][0], "video_end_pyc_s": r["video_span"][1],
        "pyc_end_s": r["pyc_span"][1],
        "resid_daq_ms": r["resid_daq_ms"], "resid_camera_ms": r["resid_camera_ms"],
        "drift_daq_ppm": r["drift_daq_ppm"], "drift_camera_ppm": r["drift_camera_ppm"],
        "fs_nominal": r["fs_nominal"], "fs_measured": r["fs_measured"],
        "trial_time_events": len(r["trial_diff"]), "trial_time_mismatches": int(np.sum(r["trial_diff"] != 0)),
        "match_r_at_offset": r["match_r"][5], "match_r_best_other": max(np.delete(r["match_r"], 5)),
        "loc_ok_A": r["loc_ok"][0], "loc_ok_B": r["loc_ok"][1], "loc_ok_C": r["loc_ok"][2]} for r in rows])
    table.to_csv(out / "sync.csv", index=False)
    print(f"Saved figure and table to {out}")


if __name__ == "__main__":
    main()
