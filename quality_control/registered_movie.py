"""
Turn a time window of a suite2p plane's registered stack into a short, sped-up mp4.

    uv run python quality_control/registered_movie.py <plane folder> [--start S] [--duration S] [options]

e.g.
    uv run python quality_control/registered_movie.py /Volumes/behrens/Sandra/mini2p/fscan/test2/preprocessed/suite2p/plane0
    uv run python quality_control/registered_movie.py /Volumes/behrens/Sandra/mini2p/fscan/test4/preprocessed/suite2p/plane1 --chan 2 --speed 5
    uv run python quality_control/registered_movie.py <plane folder> --fetch-only                # fill the cache only
    uv run python quality_control/registered_movie.py --from-cache ~/maze_mini_qc_cache/<name>.npy --speed 20

Self-contained port of Bonsai_ABCD's preprocessing/quality_control/registered_movie.py
and registered_movie_from_path.py (no import from that repo). For sanity checking
(motion, bleaching, z-drift, dead/saturated periods) and for showing data to others.
Changes from the original:
  - takes the plane folder (or data.bin itself) directly; no lab-share path
    translation and no mouse/session lookup. The movie label comes from the path
    (suite2p_io.plane_label, e.g. 'test2' or 'test4/plane1'); override with --name.
  - the default window starts at 0 s (the original's 600 s is longer than most
    Mini2P test recordings).
  - fs is read from settings.npy / ops.npy, with no 45 Hz fallback: suite2p 1.x
    stores fs=None in db.npy, which made the original fall back to 45 Hz.
  - no synthetic-stack mode.

The registered stack is suite2p's data.bin (data_chan2.bin for --chan 2): a flat,
headerless int16 array of shape (nframes, Ly, Lx). --chan follows suite2p's file
naming: 1 = data.bin = the functional channel, 2 = data_chan2.bin = the other
channel (in the Mini2P data functional_chan = 2, so --chan 1 is the green,
functional channel and --chan 2 the structural one). A time window is one
contiguous byte range, read with a single seek; no suite2p import needed.

Fetching and rendering are separate steps, because the share can be slow:
  1. fetch (slow, network): the window's frames are copied to a local cache
     (<cache_dir>/<name>.npy + <name>.json). Reads go in chunks with progress
     output; the cache is written as .part and only renamed when complete, so a
     dropped connection never leaves a half-filled cache that looks valid. Same
     parameters -> cache reused. The data folder is only ever read.
  2. render (fast, local): speed-up, crop, downsample, contrast, timestamp, mp4
     encoding. Re-running with other --speed/--clip/--crop etc. only reads the cache.
  --fetch-only / --from-cache split the two steps (e.g. across machines).

Speed-up: each output frame stands for k = round(fs * speed / fps) imaging frames,
and the output frame rate is set to fs * speed / k so the speed-up is exact.
  --mode mean (default): average blocks of k frames (also denoises the frames).
  --mode skip: take every k-th frame; only those frames are fetched.
  --bin-on-fetch: with mode mean, average (and --downsample) before caching; the
               cache is k times smaller but tied to that speed-up.

Contrast is one global percentile clip (--clip, from frames sampled across the
window) so brightness does not flicker. Encoding pipes raw frames to ffmpeg
(libx264, yuv420p: plays in browsers, Keynote, Slack); without ffmpeg it falls back
to OpenCV's mp4v. A mean projection PNG of the window is saved next to the movie.
"""
import argparse
import functools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite2p_io import DEFAULT_OUT_ROOT, plane_label

BYTES_PER_PIXEL = 2  # suite2p binaries are int16
DEFAULT_CACHE_DIR = "~/maze_mini_qc_cache"
DEFAULT_MOVIE_DIR = os.path.join(DEFAULT_OUT_ROOT, "movies")
CHAN_NAMES = {1: "functional", 2: "chan2"}


# ---------------------------------------------------------------- metadata

@functools.lru_cache(maxsize=None)
def _load_ops_cached(plane_dir):
    plane_dir = Path(plane_dir)
    db_path, settings_path, ops_path = (plane_dir / "db.npy", plane_dir / "settings.npy",
                                        plane_dir / "ops.npy")
    print(f"[share] reading suite2p metadata in {plane_dir} ...", flush=True)
    if db_path.exists():
        # suite2p 1.x: db.npy + settings.npy are much smaller than ops.npy, which
        # matters on a slow share; fs lives in settings.npy (db.npy has fs=None)
        ops = np.load(db_path, allow_pickle=True).item()
        if ops.get("fs") is None and settings_path.exists():
            ops["fs"] = np.load(settings_path, allow_pickle=True).item().get("fs")
    else:
        ops = {}
    if any(ops.get(k) is None for k in ("Ly", "Lx", "nframes", "fs")) and ops_path.exists():
        full = np.load(ops_path, allow_pickle=True).item()
        for k in ("Ly", "Lx", "nframes", "fs"):
            if ops.get(k) is None:
                ops[k] = full.get(k)
    missing = [k for k in ("Ly", "Lx", "nframes", "fs") if ops.get(k) is None]
    if missing:
        sys.exit(f"Could not find {missing} in db.npy/settings.npy/ops.npy of {plane_dir}")
    return {"Ly": int(ops["Ly"]), "Lx": int(ops["Lx"]), "nframes": int(ops["nframes"]),
            "fs": float(ops["fs"])}


def load_ops(plane_dir):
    """Ly, Lx, nframes, fs of a plane; read once per run."""
    return dict(_load_ops_cached(str(plane_dir)))


def frame_range(start_s, duration_s, fs, nframes):
    """[f0, f1) imaging frames for a window in seconds; duration 0 means to the end."""
    f0 = int(round(start_s * fs))
    if f0 >= nframes:
        sys.exit(f"--start {start_s}s is beyond the recording ({nframes / fs:.0f}s)")
    f1 = nframes if duration_s <= 0 else min(nframes, f0 + int(round(duration_s * fs)))
    return f0, f1


def frames_per_output(fs, speed, fps):
    return max(1, int(round(fs * speed / fps)))


# ---------------------------------------------------------------- fetching and cache

def _downsample(frames, factor):
    """Spatial block mean over factor x factor pixels; frames is (n, Ly, Lx)."""
    if factor <= 1:
        return frames
    n, ly, lx = frames.shape
    ly, lx = ly // factor * factor, lx // factor * factor
    f = frames[:, :ly, :lx].reshape(n, ly // factor, factor, lx // factor, factor)
    return f.mean(axis=(2, 4))


def _cache_name(name, chan, f0, f1, step, bin_k, downsample):
    out = f"{name}_{CHAN_NAMES[chan]}_f{f0}-{f1}"
    if step > 1:
        out += f"_skip{step}"
    if bin_k > 1 or downsample > 1:
        out += f"_bin{bin_k}_ds{downsample}"
    return out


def _write_cache(cache_dir, name, shape, dtype, meta, fill):
    """Stream frames into <name>.npy via fill(out_memmap); rename from .part only when complete."""
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy, part = cache_dir / f"{name}.npy", cache_dir / f"{name}.npy.part"
    out = np.lib.format.open_memmap(part, mode="w+", dtype=dtype, shape=shape)
    try:
        fill(out)
        out.flush()
        del out
    except BaseException:
        del out
        part.unlink(missing_ok=True)
        raise
    os.replace(part, npy)
    (cache_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return npy


def extract_snippet(plane_dir, name, chan=1, start_s=0, duration_s=60,
                    step=1, bin_k=1, downsample=1, cache_dir=DEFAULT_CACHE_DIR, chunk=500):
    """Copy frames [start, start+duration) of the registered binary into the local cache.

    step > 1 reads only every step-th frame (one seek each). bin_k > 1 averages blocks
    of bin_k frames, and downsample block-averages pixels, before caching (cache stored
    as float32). Returns the cache .npy path; an existing cache with the same
    parameters is reused. The binary is opened read-only.
    """
    plane_dir = Path(plane_dir)
    ops = load_ops(plane_dir)
    Ly, Lx, fs = ops["Ly"], ops["Lx"], ops["fs"]
    f0, f1 = frame_range(start_s, duration_s, fs, ops["nframes"])
    cname = _cache_name(name, chan, f0, f1, step, bin_k, downsample)
    npy = Path(cache_dir).expanduser() / f"{cname}.npy"
    if npy.exists() and npy.with_suffix(".json").exists():
        print(f"[cache] {npy}")
        return npy

    bin_path = plane_dir / ("data.bin" if chan == 1 else "data_chan2.bin")
    if not bin_path.exists():
        sys.exit(f"{bin_path} not found")
    expected = ops["nframes"] * Ly * Lx * BYTES_PER_PIXEL
    actual = os.path.getsize(bin_path)
    if actual != expected:
        print(f"[warn] {bin_path.name} is {actual} bytes, ops implies {expected} "
              f"(nframes={ops['nframes']}, Ly={Ly}, Lx={Lx}); wrong Ly/Lx or truncated file?")

    frame_bytes = Ly * Lx * BYTES_PER_PIXEL
    src_idx = np.arange(f0, f1, step)
    n_out = len(src_idx) // bin_k
    src_idx = src_idx[:n_out * bin_k]
    ly_o, lx_o = (Ly // downsample, Lx // downsample) if downsample > 1 else (Ly, Lx)
    reduce = bin_k > 1 or downsample > 1
    dtype = np.float32 if reduce else np.int16
    meta = {"source": str(bin_path), "name": name, "chan": chan, "fs": fs, "f0": f0, "f1": f1,
            "step": step, "bin_k": bin_k, "downsample": downsample,
            "frames_per_cached": step * bin_k, "Ly": ly_o, "Lx": lx_o}
    print(f"[fetch] {bin_path}\n        frames {f0}-{f1} ({(f1 - f0) / fs:.0f}s), "
          f"{len(src_idx)} frames, {len(src_idx) * frame_bytes / 1e9:.2f} GB to read")

    def fill(out):
        t0 = time.time()
        chunk_k = max(bin_k, chunk // bin_k * bin_k)  # read whole bins per chunk
        with open(bin_path, "rb") as f:
            for c0 in range(0, len(src_idx), chunk_k):
                idx = src_idx[c0:c0 + chunk_k]
                if step == 1:
                    f.seek(int(idx[0]) * frame_bytes)
                    raw = np.fromfile(f, dtype=np.int16, count=len(idx) * Ly * Lx)
                else:
                    parts = []
                    for i in idx:
                        f.seek(int(i) * frame_bytes)
                        parts.append(np.fromfile(f, dtype=np.int16, count=Ly * Lx))
                    raw = np.concatenate(parts)
                if raw.size != len(idx) * Ly * Lx:
                    raise IOError(f"short read at frame {idx[0]}: file ended or connection dropped")
                block = raw.reshape(len(idx), Ly, Lx)
                if reduce:
                    block = block.reshape(-1, bin_k, Ly, Lx).mean(axis=1, dtype=np.float32)
                    block = _downsample(block, downsample)
                o0 = c0 // bin_k
                out[o0:o0 + len(block)] = block
                done = c0 + len(idx)
                rate = done * frame_bytes / 1e6 / max(time.time() - t0, 1e-6)
                print(f"\r        {done}/{len(src_idx)} frames, {rate:.0f} MB/s", end="", flush=True)
        print()

    return _write_cache(cache_dir, cname, (n_out, ly_o, lx_o), dtype, meta, fill)


def load_cache(npy):
    npy = Path(npy).expanduser()
    meta = json.loads(npy.with_suffix(".json").read_text())
    return np.load(npy, mmap_mode="r"), meta


# ---------------------------------------------------------------- rendering (local)

def speedup(stack, k, mode="mean", crop=None, downsample=1):
    """Yield output frames (float32), one per k cached frames."""
    y0, y1, x0, x1 = crop if crop else (0, stack.shape[1], 0, stack.shape[2])
    for i in range(len(stack) // k):
        block = np.asarray(stack[i * k:(i + 1) * k, y0:y1, x0:x1], dtype=np.float32)
        frame = block.mean(axis=0) if mode == "mean" else block[0]
        yield _downsample(frame[None], downsample)[0]


def clip_limits(stack, k, mode, crop, downsample, clip=(1, 99.5), n_sample=200, seed=0):
    """Global intensity limits from output frames sampled across the whole window."""
    n_out = len(stack) // k
    idx = np.sort(np.random.default_rng(seed).choice(n_out, min(n_sample, n_out), replace=False))
    y0, y1, x0, x1 = crop if crop else (0, stack.shape[1], 0, stack.shape[2])
    frames = []
    for i in idx:
        block = np.asarray(stack[i * k:(i + 1) * k, y0:y1, x0:x1], dtype=np.float32)
        frames.append(_downsample((block.mean(axis=0) if mode == "mean" else block[0])[None], downsample)[0])
    lo, hi = np.percentile(np.stack(frames), clip)
    return float(lo), float(hi)


def normalise(frame, lo, hi, gamma=1.0):
    x = np.clip((frame - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        x = x ** gamma
    return (x * 255).astype(np.uint8)


def annotate(img, text_left, text_right):
    """Burn white-on-black-outline text into the bottom corners of a uint8 frame."""
    h, w = img.shape
    scale = max(0.35, w / 900)
    thick = max(1, int(round(scale * 2)))
    for text, align in ((text_left, "left"), (text_right, "right")):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (6, h - 8) if align == "left" else (w - tw - 6, h - 8)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
    return img


def write_mp4(frames, out_path, fps, size, crf=23):
    """Encode an iterable of uint8 (h, w) frames; ffmpeg/libx264 if available, else OpenCV mp4v."""
    w, h = size
    ffmpeg = shutil.which("ffmpeg")
    n = 0
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
               "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium",
               "-movflags", "+faststart", str(out_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for img in frames:
                proc.stdin.write(np.ascontiguousarray(img).tobytes())
                n += 1
        finally:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")
    else:
        print("[warn] ffmpeg not found, falling back to OpenCV mp4v (larger files, may not play in browsers)")
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
        for img in frames:
            vw.write(img)
            n += 1
        vw.release()
    return n


def _fmt_time(s):
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def render(npy, out_dir=DEFAULT_MOVIE_DIR, speed=10, fps=30, mode="mean", crop=None,
           downsample=1, clip=(1, 99.5), gamma=1.0, crf=23):
    """Render a cached snippet to mp4 + mean projection PNG; returns the mp4 path."""
    stack, meta = load_cache(npy)
    fs_cached = meta["fs"] / meta["frames_per_cached"]
    k = frames_per_output(fs_cached, speed, fps)
    out_fps = fs_cached * speed / k
    if crop:
        crop = [int(c) for c in crop]
    # libx264 + yuv420p need even dimensions: trim the crop by a pixel if needed
    y0, y1, x0, x1 = crop if crop else (0, stack.shape[1], 0, stack.shape[2])
    h, w = (y1 - y0) // downsample, (x1 - x0) // downsample
    y1 -= (h % 2) * downsample
    x1 -= (w % 2) * downsample
    crop = (y0, y1, x0, x1)
    h, w = h - h % 2, w - w % 2

    lo, hi = clip_limits(stack, k, mode, crop, downsample, clip)
    n_out = len(stack) // k
    t_rec = lambda i: (meta["f0"] + i * k * meta["frames_per_cached"]) / meta["fs"]
    label = f"{meta['name'].replace('_', ' ')} {CHAN_NAMES[meta['chan']]}"
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0s, t1s = meta["f0"] / meta["fs"], meta["f1"] / meta["fs"]
    stem = f"{meta['name']}_{CHAN_NAMES[meta['chan']]}_{t0s:.0f}-{t1s:.0f}s_x{speed:g}_{mode}"
    out_path = out_dir / f"{stem}.mp4"
    print(f"[render] {n_out} output frames: {k} cached frame(s) each ({mode}), "
          f"{out_fps:.2f} fps -> x{speed:g}, {w}x{h}, clip {lo:.0f}-{hi:.0f}")

    total = np.zeros((h, w), np.float64)

    def frames():
        for i, frame in enumerate(speedup(stack, k, mode, crop, downsample)):
            total[:] += frame[:h, :w]
            img = normalise(frame[:h, :w], lo, hi, gamma)
            yield annotate(img, label, f"{_fmt_time(t_rec(i))}  x{speed:g}")
            if i % 50 == 0:
                print(f"\r        {i}/{n_out}", end="", flush=True)

    n = write_mp4(frames(), out_path, out_fps, (w, h), crf)
    print(f"\r        {n}/{n_out}")
    mean_img = normalise(total / max(n, 1), *np.percentile(total / max(n, 1), clip))
    cv2.imwrite(str(out_dir / f"{stem}_mean.png"), mean_img)
    print(f"[saved] {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB, {n / out_fps:.1f}s)")
    return out_path


# ---------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plane_dir", nargs="?", help="suite2p plane folder with data.bin (or data.bin itself)")
    p.add_argument("--name", help="label for movie/cache names (default: from the path, e.g. test4_plane1)")
    p.add_argument("--start", type=float, default=0, help="window start, s (default 0)")
    p.add_argument("--duration", type=float, default=60, help="window length, s; 0 = to end (default 60)")
    p.add_argument("--chan", type=int, choices=[1, 2], default=1,
                   help="1 = data.bin (functional), 2 = data_chan2.bin (default 1)")
    p.add_argument("--speed", type=float, default=10, help="playback speed-up (default 10)")
    p.add_argument("--fps", type=float, default=30, help="target output frame rate (default 30)")
    p.add_argument("--mode", choices=["mean", "skip"], default="mean")
    p.add_argument("--bin-on-fetch", action="store_true", help="mode mean: average/downsample before caching")
    p.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    p.add_argument("--downsample", type=int, default=1, help="spatial block-mean factor")
    p.add_argument("--clip", type=float, nargs=2, default=(1, 99.5), metavar=("LO", "HI"),
                   help="intensity percentiles mapped to black/white (default 1 99.5)")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--crf", type=int, default=23, help="x264 quality, lower = better/larger (default 23)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--out", default=DEFAULT_MOVIE_DIR)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fetch-only", action="store_true", help="only fill the cache")
    g.add_argument("--from-cache", help="render from a cached .npy (with its .json) instead of fetching")
    a = p.parse_args()

    render_downsample = a.downsample
    if a.from_cache:
        npy = a.from_cache
    else:
        if not a.plane_dir:
            p.error("give plane_dir (or --from-cache)")
        plane = Path(a.plane_dir).expanduser()
        if plane.is_file():
            plane = plane.parent
        if not plane.is_dir():
            sys.exit(f"Folder not found: {plane}")
        plane = plane.resolve()
        name = a.name or plane_label(str(plane), str(plane)).replace("/", "_")
        fs = load_ops(plane)["fs"]
        k = frames_per_output(fs, a.speed, a.fps)
        step = k if a.mode == "skip" else 1
        bin_k = k if (a.mode == "mean" and a.bin_on_fetch) else 1
        fetch_ds = a.downsample if a.bin_on_fetch else 1
        npy = extract_snippet(plane, name, a.chan, a.start, a.duration,
                              step=step, bin_k=bin_k, downsample=fetch_ds, cache_dir=a.cache_dir)
        if fetch_ds > 1:
            render_downsample = 1
    if a.fetch_only:
        print(f"[done] cached at {npy} (+ .json); render with --from-cache")
        return
    render(npy, a.out, a.speed, a.fps, a.mode, a.crop, render_downsample, tuple(a.clip), a.gamma, a.crf)


if __name__ == "__main__":
    main()
