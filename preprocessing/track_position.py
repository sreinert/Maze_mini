"""
Low-effort position tracking of the mouse in the maze videos, from background
subtraction (as in old_testing/tracking_test.ipynb), with QC figures to check
that the trajectories are free of jumps.

    uv run python preprocessing/track_position.py /Volumes/behrens/Sandra/mini2p/Arya/RAW_GOOD
    uv run python preprocessing/track_position.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding --session 2026-08-13

Takes an ABC root (raw layout, abc_data.raw, or Arya's processed layout) and
tracks the video of every session. On the processed layout the result is also
compared with Arya's XY (head position from her tracking, aligned frames).

Tracker, per video frame (grey, at half resolution):
    background  per-pixel median of --n-bg frames spread over the session (the
                mouse moves, so it drops out of the median)
    mouse       the mouse is darker than the background; the cable, headstage
                and port LEDs are bright, so only the darkening is used (the
                notebook's |frame - mean| argmax locks onto the cable), relative
                to the background, (bg - frame) / (bg + 4): on the dim bridges
                (over the dark floor) the absolute darkening is tiny and the
                track would jump from platform to platform.
                Blurred (sigma 3 px), thresholded at --threshold, closed with
                a disk of radius --close-px (joins head and body, which the
                bright headstage splits) and opened with a disk of radius
                --open-px, which removes the thin
                dark cable (where it crosses the bright platforms) and tail but
                keeps the body. Blob: the largest one, or, if several are at
                least half its size, the one nearest the previous position.
                Point: centroid of the blob (body centre). Not the darkening
                peak: that sits on whichever end of the mouse is on a bright
                platform and flips across the bridge.
    lost        blob smaller than --min-area (px, full resolution) -> NaN
Location (abc_data.maze.classify_positions): node if within 0.3 node
spacings of its port LED (found in the background), else the nearest bridge.
On the processed layout it is compared with Arya's maze_location.
Cleaning:
    outliers    frames farther than --jump-px from the running median (window
                --median-s) of the raw track -> NaN
    gaps        NaN runs up to --max-gap-s are linearly interpolated; longer
                gaps stay NaN
The point is the body centre, not the head: expect a ~20-40 px offset from
Arya's head XY.

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of root>/tracking;
nothing is written next to the data):
    <session>_tracking.csv     per video frame: time (camera s), x, y (cleaned,
                               full-res video px), x_raw, y_raw, area, status
                               (ok / lost / outlier / interpolated)
    <session>_imaging_xy.npy   (n imaging frames, 2) mean cleaned position per
                               suite2p frame (DAQ frame clock -> pyControl ->
                               camera), NaN without tracking
    <session>_imaging_location.npy  (n imaging frames,) location id per suite2p
                               frame (abc_data.maze ids: nodes 1-9, edges 10-21,
                               -1 unknown), from imaging_xy
    <session>_node_centres.npy (10, 2) port LED per node (row = node id), found
                               in the background (maze.node_centres_from_background)
    <session>_tracking.png     trajectory over the background, x/y vs time,
                               speed, step sizes before/after cleaning,
                               positions at A/B/C pokes (should sit on 3
                               ports), (processed layout) distance to Arya's XY
    <session>_largest_steps.png  video crops around the largest remaining
                               frame-to-frame steps, before and after
    summary.csv
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quality_control"))

import cv2  # noqa: E402

from abc_common import GOAL_COLORS, GOAL_NAMES, labels, new_figure, safe, save, title  # noqa: E402
from abc_common import parser, setup  # noqa: E402
from abc_data import maze  # noqa: E402
from abc_data.sync import time_to_frame  # noqa: E402
from suite2p_io import INK_MUTED, INK_SECONDARY, SERIES_1, SERIES_2, colormap, style_axis  # noqa: E402

SCALE = 2  # work at 1 / SCALE resolution


def _grey(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[::SCALE, ::SCALE].astype(np.float32)


def background(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    imgs = []
    for k in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, fr = cap.read()
        if ok:
            imgs.append(_grey(fr))
    cap.release()
    return np.median(imgs, axis=0).astype(np.float32), total


def track_video(path, bg, threshold, min_area, open_px, close_px):
    """(n_frames, 2) raw position (full-res px, NaN = lost) and (n_frames,) blob area."""
    cap = cv2.VideoCapture(path)
    r = max(int(round(open_px / SCALE)), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    rc = max(int(round(close_px / SCALE)), 1)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rc + 1, 2 * rc + 1))
    xy, area = [], []
    prev = None
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        d = np.clip((bg - _grey(fr)) / (bg + 4), 0, None)
        mask = (cv2.GaussianBlur(d, (0, 0), 3) > threshold).astype(np.uint8)
        # closing joins head and body across the bright headstage; opening then
        # removes thin dark structures (cable, tail) and keeps the body
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close), cv2.MORPH_OPEN, kernel)
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
        areas = st[1:, cv2.CC_STAT_AREA] * SCALE ** 2
        if n < 2 or areas.max() < min_area:
            xy.append((np.nan, np.nan))
            area.append(areas.max() if n > 1 else 0)
            continue
        # among blobs at least half as large as the largest, the one nearest the last position
        cand = np.flatnonzero(areas >= 0.5 * areas.max())
        if prev is not None and len(cand) > 1:
            i = 1 + cand[np.argmin(np.hypot(*(cen[1 + cand] * SCALE - prev).T))]
        else:
            i = 1 + int(np.argmax(areas))
        area.append(areas[i - 1])
        prev = (cen[i] + 0.5) * SCALE
        xy.append(tuple(prev))
        if len(xy) % 10000 == 0:
            print(f"    {len(xy)} frames, {len(xy) / (time.time() - t0):.0f} fps", flush=True)
    cap.release()
    return np.array(xy, float), np.array(area)


def clean(xy, fps, jump_px, median_s, max_gap_s):
    """Cleaned track, status per frame (0 ok, 1 lost, 2 outlier, 3 interpolated)
    and the status before interpolation (0 ok, 1 lost, 2 outlier)."""
    status = np.where(np.isfinite(xy[:, 0]), 0, 1)
    w = max(int(round(median_s * fps)) | 1, 3)
    med = pd.DataFrame(xy).rolling(w, center=True, min_periods=w // 2 + 1).median().to_numpy()
    dev = np.hypot(*(xy - med).T)
    status[(status == 0) & (dev > jump_px)] = 2
    pre = status.copy()
    out = np.where(status[:, None] == 0, xy, np.nan)
    # interpolate short gaps only
    bad = np.isnan(out[:, 0])
    edges = np.diff(np.r_[0, bad.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    t = np.arange(len(out))
    good = ~bad
    for a, b in zip(starts, ends):
        if a == 0 or b == len(out) or (b - a) > max_gap_s * fps:
            continue
        for c in (0, 1):
            out[a:b, c] = np.interp(t[a:b], t[good], out[good, c])
        status[a:b] = 3
    return out, status, pre


def steps(xy):
    """Frame-to-frame step (px); NaN across missing frames."""
    return np.r_[np.nan, np.hypot(*np.diff(xy, axis=0).T)]


def imaging_xy(s, xy):
    """(n imaging frames, 2) mean position over the video frames inside each
    suite2p frame."""
    ft = s.imaging_sync.frame_times
    pyc = (np.r_[ft, ft[-1] + np.median(np.diff(ft))] - s.pyc_to_daq.intercept) / s.pyc_to_daq.slope
    cam = time_to_frame(s.pyc_to_camera(pyc), s.camera_sync.frame_times)
    n_cam = min(len(xy), len(s.camera_sync.frame_times))
    out = np.full((len(ft), 2), np.nan)
    for i in range(len(ft)):
        a, b = max(cam[i], 0), min(cam[i + 1], n_cam)
        if b > a:
            seg = xy[a:b]
            if np.isfinite(seg[:, 0]).any():
                out[i] = np.nanmean(seg, axis=0)
    return out


def goal_camera_frames(s):
    out = []
    for n in ("A_on", "B_on", "C_on"):
        k = time_to_frame(s.pyc_to_camera(s.pycontrol.state_times(n)), s.camera_sync.frame_times)
        out.append(k[k >= 0])
    return out


def arya_comparison(s, xy_img):
    """Distance (px) between our per-imaging-frame position and Arya's XY on the
    aligned frames (NaN for raw sessions)."""
    if not hasattr(s, "xy"):
        return None
    ours = xy_img[s.aligned_offset:s.aligned_offset + s.n_frames]
    theirs = np.asarray(s.xy, float)[:len(ours)]
    return np.hypot(*(ours - theirs).T)


def arya_location_agreement(s, loc_img):
    """Fraction of aligned frames (both labelled) where our location equals
    Arya's maze_location (None for raw sessions)."""
    if not hasattr(s, "location"):
        return None
    ours = loc_img[s.aligned_offset:s.aligned_offset + s.n_frames]
    theirs = s.location[:len(ours)]
    ok = (ours != maze.UNKNOWN) & (theirs != maze.UNKNOWN)
    return float(np.mean(ours[ok] == theirs[ok]))


def plot_session(s, bg, xy_raw, xy, pre, area, fps, a, dist, centres, loc_img, agree, out_path):
    fig, axes = new_figure(2, 3, figsize=(20, 11))
    ax = axes.ravel()
    for x in ax:
        style_axis(x)
    t = np.arange(len(xy)) / fps
    H, W = bg.shape[0] * SCALE, bg.shape[1] * SCALE

    x = ax[0]
    x.imshow(bg, cmap="gray", extent=[0, W, H, 0])
    sc = x.scatter(xy[:, 0], xy[:, 1], c=t, s=0.3, cmap=colormap(), alpha=0.5, edgecolor="none")
    x.plot(xy[:, 0], xy[:, 1], color=SERIES_2, linewidth=0.15, alpha=0.4)
    fig.colorbar(sc, ax=x, shrink=0.7, label="time (s)")
    for n in maze.NODES:
        x.text(*centres[n], str(n), color="white", fontsize=9, fontweight="bold", ha="center", va="center")
    x.set_xlim(0, W)
    x.set_ylim(H, 0)
    title(x, "Cleaned trajectory over the background (node ids at the port LEDs)")
    labels(x, "x (video px)", "y (video px)")

    x = ax[1]
    for c, (name, col) in enumerate((("x", SERIES_1), ("y", SERIES_2))):
        x.plot(t, xy_raw[:, c], color=INK_MUTED, linewidth=0.4, label="raw" if c == 0 else None)
        x.plot(t, xy[:, c], color=col, linewidth=0.6, label=f"{name} cleaned")
    for code, nm, col in ((1, "lost", INK_SECONDARY), (2, "outlier", SERIES_2)):
        m = pre == code
        x.plot(t[m], np.full(m.sum(), -30 - 30 * code), "|", color=col, markersize=4,
               label=f"{nm} ({m.mean():.1%})")
    x.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    title(x, "Position vs time")
    labels(x, "time (s)", "video px")

    x = ax[2]
    sp = steps(xy) * fps
    x.plot(t, sp, color=SERIES_1, linewidth=0.4)
    x.axhline(a.jump_px * fps, color=INK_MUTED, linestyle="--", linewidth=1, label="--jump-px per frame")
    x.set_ylim(0, max(np.nanpercentile(sp, 99.95) * 1.2, a.jump_px * fps * 1.1))
    x.legend(frameon=False, fontsize=8)
    title(x, "Speed (cleaned)")
    labels(x, "time (s)", "px/s")

    x = ax[3]
    bins = np.logspace(-1, 3, 60)
    for v, lab_, col in ((steps(xy_raw), "raw", INK_MUTED), (steps(xy), "cleaned", SERIES_1)):
        v = v[np.isfinite(v)]
        x.hist(np.clip(v, 0.1, None), bins=bins, histtype="step", color=col, linewidth=1.5,
               label=f"{lab_}: max {v.max():.0f} px, > {a.jump_px:g} px: {np.sum(v > a.jump_px)}")
    x.axvline(a.jump_px, color=INK_MUTED, linestyle="--", linewidth=1)
    x.set_xscale("log")
    x.set_yscale("log")
    x.legend(frameon=False, fontsize=8)
    title(x, "Frame-to-frame steps")
    labels(x, "step (px)", "frames")

    x = ax[4]
    x.imshow(bg, cmap="gray", extent=[0, W, H, 0])
    for frames, name, col in zip(goal_camera_frames(s), GOAL_NAMES, GOAL_COLORS):
        frames = frames[frames < len(xy)]
        x.scatter(xy[frames, 0], xy[frames, 1], s=18, color=col, edgecolor="white", linewidth=0.5,
                  label=f"{name}_on ({len(frames)})")
    x.set_xlim(0, W)
    x.set_ylim(H, 0)
    x.legend(frameon=False, fontsize=8, labelcolor="white")
    title(x, f"Position at goal pokes (task {'-'.join(map(str, s.task))})")

    x = ax[5]
    if dist is not None and np.isfinite(dist).any():
        d = dist[np.isfinite(dist)]
        x.hist(d, bins=np.linspace(0, 200, 81), color=SERIES_1)
        title(x, f"Distance to Arya's XY per imaging frame\nmedian {np.median(d):.0f} px, "
                 f"> 100 px: {np.mean(d > 100):.1%}; location = hers in {agree:.1%}")
        labels(x, "distance (px)", "imaging frames")
    else:
        ids = [maze.UNKNOWN] + [l for l in maze.LOCATIONS if l not in maze.ABSENT_EDGES]
        occ = [np.mean(loc_img == l) for l in ids]
        x.bar(range(len(ids)), occ, color=[INK_MUTED] + [SERIES_1 if l in maze.NODES else SERIES_2 for l in ids[1:]],
              edgecolor="white")
        x.set_xticks(range(len(ids)))
        x.set_xticklabels(["?"] + [maze.location_name(l) for l in ids[1:]], rotation=90, fontsize=7.5)
        x.plot([], [], "s", color=SERIES_1, label="nodes")
        x.plot([], [], "s", color=SERIES_2, label="edges")
        x.legend(frameon=False, fontsize=8)
        title(x, "Location occupancy (imaging frames)")
        labels(x, y="fraction of frames")
    good = np.isfinite(xy[:, 0]).mean()
    save(fig, out_path, f"{s.label}  |  {os.path.basename(s.video_path)}  |  {len(xy)} frames at {fps:.0f} Hz, "
                        f"{good:.1%} tracked after cleaning")


def plot_largest_steps(path, xy, n=8, half=90):
    """Crops (before | after) around the n largest steps of the cleaned track."""
    st = steps(xy)
    order = np.argsort(np.nan_to_num(st, nan=-1))[::-1][:n]
    cap = cv2.VideoCapture(path)
    fig, axes = new_figure(2, n, figsize=(2.2 * n, 5), squeeze=False)
    for j, k in enumerate(sorted(order)):
        for r, f in enumerate((k - 1, k)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, fr = cap.read()
            axx = axes[r, j]
            axx.axis("off")
            if not ok:
                continue
            cx, cy = np.nan_to_num(xy[k], nan=0).astype(int)
            y0, x0 = max(cy - half, 0), max(cx - half, 0)
            axx.imshow(fr[y0:cy + half, x0:cx + half, ::-1])
            for ff, col in ((k - 1, SERIES_1), (k, SERIES_2)):
                axx.plot(xy[ff, 0] - x0, xy[ff, 1] - y0, "+", color=col, markersize=10, markeredgewidth=2)
            title(axx, f"frame {f}" + (f"\nstep {st[k]:.0f} px" if r == 0 else ""))
    cap.release()
    return fig


def main():
    p = parser(__doc__, "tracking")
    p.add_argument("--n-bg", type=int, default=150, help="frames for the median background (default 150)")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="threshold on the relative darkening (bg - frame) / bg (default 0.15)")
    p.add_argument("--min-area", type=float, default=1500, help="min blob area, full-res px (default 1500)")
    p.add_argument("--open-px", type=float, default=14,
                   help="radius of the opening that removes cable and tail, full-res px (default 14)")
    p.add_argument("--close-px", type=float, default=16,
                   help="radius of the closing that joins head and body across the headstage, full-res px (default 16)")
    p.add_argument("--jump-px", type=float, default=40, help="outlier distance from running median, px (default 40)")
    p.add_argument("--median-s", type=float, default=0.25, help="running-median window, s (default 0.25)")
    p.add_argument("--max-gap-s", type=float, default=0.5, help="longest gap interpolated, s (default 0.5)")
    a = p.parse_args()
    root, out, sessions = setup(a, "tracking")
    rows = []
    for s in sessions:
        if not s.video_path:
            print(f"{s.label}: no video, skipped")
            continue
        print(f"{s.label}: {s.video_path}", flush=True)
        bg, total = background(s.video_path, a.n_bg)
        fps = 1 / np.median(s.camera_sync.frame_intervals)
        xy_raw, area = track_video(s.video_path, bg, a.threshold, a.min_area, a.open_px, a.close_px)
        xy, status, pre = clean(xy_raw, fps, a.jump_px, a.median_s, a.max_gap_s)
        cam_t = s.camera_sync.frame_times
        t = np.r_[cam_t, cam_t[-1] + np.arange(1, len(xy) - len(cam_t) + 1) / fps][:len(xy)]
        stem = safe(s.label)
        centres = maze.node_centres_from_background(bg, SCALE)
        pd.DataFrame({"frame": np.arange(len(xy)), "time_s": t, "x": xy[:, 0], "y": xy[:, 1],
                      "x_raw": xy_raw[:, 0], "y_raw": xy_raw[:, 1], "area": area,
                      "location": maze.classify_positions(xy, centres),
                      "status": np.array(["ok", "lost", "outlier", "interpolated"])[status]}
                     ).to_csv(out / f"{stem}_tracking.csv", index=False, float_format="%.4f")
        xy_img = imaging_xy(s, xy)
        np.save(out / f"{stem}_imaging_xy.npy", xy_img)
        np.save(out / f"{stem}_node_centres.npy", centres)
        loc_img = maze.classify_positions(xy_img, centres)
        np.save(out / f"{stem}_imaging_location.npy", loc_img)
        agree = arya_location_agreement(s, loc_img)
        dist = arya_comparison(s, xy_img)
        plot_session(s, bg, xy_raw, xy, pre, area, fps, a, dist, centres, loc_img, agree, out / f"{stem}_tracking.png")
        save(plot_largest_steps(s.video_path, xy), out / f"{stem}_largest_steps.png",
             f"{s.label}: largest remaining steps (blue + = previous frame, orange + = this frame)")
        st = steps(xy)
        row = {"session": s.label, "video_frames": len(xy), "gpio_rows": len(cam_t), "fps": fps,
               "lost": np.mean(pre == 1), "outlier": np.mean(pre == 2),
               "interpolated": np.mean(status == 3), "tracked": np.isfinite(xy[:, 0]).mean(),
               "max_step_px": np.nanmax(st), "steps_over_jump": int(np.nansum(st > a.jump_px)),
               "imaging_frames_tracked": np.isfinite(xy_img[:, 0]).mean()}
        row["imaging_frames_on_node"] = float(np.mean(maze.is_node(loc_img)))
        row["imaging_frames_unknown"] = float(np.mean(loc_img == maze.UNKNOWN))
        if dist is not None:
            row["median_dist_arya_px"] = np.nanmedian(dist)
            row["location_agreement_arya"] = agree
        rows.append(row)
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
