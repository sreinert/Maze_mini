"""
Inventory and consistency table for a preprocessed ABC-task folder: one row per
session, with every count that has to agree across files.

    uv run python quality_control/abc_inventory.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding

Checked per session (a failed check is listed in the 'flags' column):

    files       every file kind present (sessions.find_sessions)
    aligned     activity, XY, location share one length T; trial_times end at T;
                n_trials = number of pyControl C_on; task = pyControl active_poke
    imaging     suite2p frames = h5 FrameClock edges (up to 2 trailing frames
                missing is accepted and noted; see abc_data/sync.py); frame
                intervals regular (max/min within 2 % of the median);
                green_channel.bin size = 512 x 512 x frames x int16
    sync        rsync pulses: h5 PyControlPulse = pyControl rsync = camera GPIO1
    video       mp4 frame count (decoded when the container count disagrees)
                = GPIO rows; camera intervals regular (no dropped frames: none
                > 1.5 x median). Video frame i is taken to be GPIO row i.
    ROIs        suite2p ROIs = Cellpose masks (chan1 + chan2 seg files) and
                roi_table rows; ROICaT cells present <= suite2p ROIs
    offset      aligned window start from the trial times agrees with every
                event (offset_votes unanimous)

Output (in --out, default ~/Desktop/maze_mini_qc/<name of root>/inventory):
inventory.csv. Inputs are opened read-only.
"""
import os

import numpy as np
import pandas as pd

from abc_common import parser, setup


def _video_frames(path, expected=None):
    """Frame count, fps, width, height. The container's frame count is trusted only
    if it equals expected; otherwise the frames are counted by decoding."""
    import cv2
    c = cv2.VideoCapture(path)
    n, fps = int(c.get(cv2.CAP_PROP_FRAME_COUNT)), c.get(cv2.CAP_PROP_FPS)
    w, h = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if expected is not None and n != expected:
        n = 0
        while c.grab():
            n += 1
    c.release()
    return n, fps, w, h


def _seg_count(path):
    if path is None:
        return 0
    return int(np.load(path, allow_pickle=True).item()["masks"].max())


def inventory_row(s):
    flags = []

    def check(ok, msg):
        if not ok:
            flags.append(msg)

    f = s.files.files
    missing = s.files.missing()
    src = np.asarray(s.s2p("roi_source_channel"))
    if "seg_chan2" in missing and not np.any(src == "chan2"):
        missing.remove("seg_chan2")  # no chan2 ROIs, so no chan2 segmentation needed
    check(not missing, f"missing {missing}")
    T = s.n_frames
    log = s.pycontrol
    n_c = len(log.state_times("C_on"))
    r = {"session": s.label, "tag": s.tag, "roicat_session": s.files.roicat_session,
         "task_name": log.info.get("task_name"), "pyc_start": log.info.get("start_time"),
         "pyc_duration_s": round(log.duration, 1), "task_ABC": "-".join(map(str, s.task)),
         "T_activity": T, "T_xy": len(s.xy), "T_location": len(s.location),
         "trial_times_end": int(s.trial_times[-1, 3]), "n_trials": len(s.trial_times), "n_C_on": n_c,
         "n_A_on": len(log.state_times("A_on"))}
    check(r["T_xy"] == T and r["T_location"] == T, "aligned arrays differ in length")
    check(r["trial_times_end"] == T, "trial_times do not end at T")
    check(len(s.trial_times) == n_c, "n_trials != n C_on")
    check(list(s.task) == log.active_poke, f"task {list(s.task)} != active_poke {log.active_poke}")

    sy = s.imaging_sync
    ifi = np.diff(sy.frame_times[:sy.n_edges])
    r.update({"s2p_frames": s.n_s2p_frames, "h5_frame_edges": sy.n_edges,
              "frames_extrapolated": sy.n_extrapolated, "fs_nominal": s.fs,
              "fs_measured": round(s.fs_measured, 4),
              "ifi_min_ms": round(1e3 * ifi.min(), 2), "ifi_max_ms": round(1e3 * ifi.max(), 2)})
    check(s.n_s2p_frames - sy.n_edges in (0, 1, 2), "suite2p frames != h5 frame edges")
    check(ifi.max() < 1.02 * np.median(ifi) and ifi.min() > 0.98 * np.median(ifi), "irregular imaging frames")
    bin_frames = os.path.getsize(s.movie_path) / (512 * 512 * 2)
    r["bin_frames"] = bin_frames
    check(bin_frames == s.n_s2p_frames, "green_channel.bin size != 512x512 x frames")

    cam = s.camera_sync
    cifi = cam.frame_intervals
    n_vid, fps, w, h = _video_frames(s.video_path, expected=len(cam.frame_times))
    r.update({"pulses_h5": len(sy.pulse_times), "pulses_pyc": len(log.rsync_times()),
              "pulses_camera": len(cam.pulse_times),
              "video_frames": n_vid, "gpio_rows": len(cam.frame_times), "video_fps": fps,
              "video_size": f"{w}x{h}", "camera_fps_measured": round(1 / np.median(cifi), 3),
              "camera_dropped": int(np.sum(cifi > 1.5 * np.median(cifi))),
              "sync_resid_daq_ms": round(1e3 * s.pyc_to_daq.max_abs_residual, 2),
              "sync_resid_camera_ms": round(1e3 * s.pyc_to_camera.max_abs_residual, 2)})
    check(r["pulses_h5"] == r["pulses_pyc"] == r["pulses_camera"], "rsync counts differ")
    r["video_extra_frames"] = n_vid - len(cam.frame_times)
    check(n_vid == len(cam.frame_times), f"video has {n_vid - len(cam.frame_times):+d} frames vs GPIO rows")
    check(r["camera_dropped"] == 0, f"{r['camera_dropped']} dropped camera frames")

    n_roi = s.s2p("F").shape[0]
    r.update({"n_roi": n_roi, "roi_table_rows": len(s.roi_table),
              "roi_chan1": int(np.sum(src == "chan1")), "roi_chan2": int(np.sum(src == "chan2")),
              "cellpose_chan1": _seg_count(f["seg_chan1"]), "cellpose_chan2": _seg_count(f["seg_chan2"]),
              "roicat_present": int(s.present.sum()), "roicat_total": s.activity.shape[1]})
    check(r["roi_table_rows"] == n_roi, "roi_table rows != ROIs")
    check(r["roicat_present"] <= n_roi, "more ROICaT cells than suite2p ROIs")

    votes = s.offset_votes
    r.update({"aligned_offset": s.aligned_offset, "offset_at_pyc_start": s.offset_at_pyc_start,
              "offset_votes_agree": float(np.mean(votes == s.aligned_offset)),
              "window_end_pyc_s": round(float(s.time[-1]), 1)})
    check(r["offset_votes_agree"] == 1, "trial times disagree on the window offset")
    r["flags"] = "; ".join(flags)
    return r


def main():
    p = parser(__doc__, "inventory")
    a = p.parse_args()
    root, out, sessions = setup(a, "inventory")
    rows = []
    for s in sessions:
        r = inventory_row(s)
        rows.append(r)
        print(f"{s.label}: T={r['T_activity']}, {r['n_trials']} trials, {r['roicat_present']} cells, "
              f"flags: {r['flags'] or 'none'}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "inventory.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.set_index("session").T.to_string())
    print(f"Saved {out / 'inventory.csv'}")


if __name__ == "__main__":
    main()
