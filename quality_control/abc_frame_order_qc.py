"""
Frame order of Arya's processed movies vs the raw acquisition, without
matching cells: our suite2p run on the raw tif
(<root>/raw_funcimg/<tag>/preprocessed/suite2p/plane0) and Arya's extraction
(<root>/suite2p_output/<tag>_bidicorrected_denoised_registered) see the same
field of view, so a whole-FOV signal (mean Fneu over all ROIs, i.e. the
neuropil; or, when our run is registration only, the frame mean of the
registered movies data.bin / green_channel.bin) must agree frame by frame if
the frames are in the same order.

    uv run python quality_control/abc_frame_order_qc.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding

Per session:
    map         every --window-frame window of Arya's signal (fast fluctuations
                only, 31-frame running mean removed) matched to its best
                window in ours: her frame -> raw frame. A correct movie lies on
                the diagonal.
    block fit   the map is described as one block of raw frames
                [start, start + n) cut out and reinserted so that it begins at
                her frame `insert` (found in all 6 good_decoding sessions:
                raw 500-999 moved to the end, in 08-31 to her frame ~5000);
                start, n and insert are refined on a +-12 frame grid by the
                correlation of the reordered signal with hers. The fitted
                boundaries are good to a few frames (whole-FOV signal, 9-frame
                high-pass).
    xcorr       correlation of the two signals vs constant shift (lag L:
                her frame k vs raw frame k + L).

Outputs (--out, default ~/Desktop/maze_mini_qc/<name of root>/frame_order):
frame_order.png (row per session: map, xcorr), frame_order.csv (fit and
correlations: identity, best constant shift, fitted reorder),
<session>_her_to_raw.npy (raw frame of each of Arya's frames under the fit;
undo the reorder of any (.., n_frames) array of hers with
x_fixed[..., her_to_raw] = x), <session>_signals.npy (the two signals, cache).
Inputs are opened read-only.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abc_common import labels, new_figure, parser, safe, save, title
from abc_data import find_sessions
from suite2p_io import GRIDLINE, INK_MUTED, SERIES_1, SERIES_2, default_out_dir, style_axis


def frame_means(path, n_frames):
    """(n_frames,) mean of every 7th pixel of each frame of an int16 movie."""
    m = np.memmap(path, np.int16, "r", shape=(n_frames, os.path.getsize(path) // 2 // n_frames))
    return np.array([m[i, ::7].astype(np.float64).mean() for i in range(n_frames)])


def signals(ours, hers, cache):
    """(ours, hers, kind) whole-FOV signals of the two runs."""
    if cache.exists():
        a, b = np.load(cache)
        kind = "mean Fneu" if (ours / "Fneu.npy").exists() else "frame mean"
        return a, b, kind
    T = np.load(hers / "Fneu.npy", mmap_mode="r").shape[1]
    if (ours / "Fneu.npy").exists():
        a = np.asarray(np.load(ours / "Fneu.npy", mmap_mode="r"), np.float64).mean(0)
        b = np.asarray(np.load(hers / "Fneu.npy", mmap_mode="r"), np.float64).mean(0)
        kind = "mean Fneu"
    else:
        a, b = frame_means(ours / "data.bin", T), frame_means(hers / "green_channel.bin", T)
        kind = "frame mean"
    np.save(cache, np.vstack([a, b]))
    return a, b, kind


def highpass(x, w):
    return x - np.convolve(x, np.ones(w) / w, "same")


def window_map(A, B, w, step):
    """(n, 3) rows: her window start, best-matching raw window start, r."""
    Aw = np.lib.stride_tricks.sliding_window_view(A, w)
    Aw = (Aw - Aw.mean(1, keepdims=True)) / Aw.std(1, keepdims=True)
    rows = []
    for s0 in range(16, len(B) - w - 16, step):
        seg = (B[s0:s0 + w] - B[s0:s0 + w].mean()) / B[s0:s0 + w].std()
        r = Aw @ seg / w
        rows.append((s0, int(r.argmax()), r.max()))
    return np.array(rows)


def reorder(T, start, n, insert):
    """her frame -> raw frame when raw[start:start + n] was moved to her frame insert."""
    rest = np.r_[np.arange(start), np.arange(start + n, T)]
    return np.r_[rest[:insert], np.arange(start, start + n), rest[insert:]]


def longest_run(mask):
    """Boolean mask of the longest run of consecutive True in mask."""
    best, cur, out = (0, 0), None, np.zeros_like(mask)
    for i, m in enumerate(np.r_[mask, False]):
        if m and cur is None:
            cur = i
        elif not m and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    out[best[0]:best[1]] = True
    return out


def fit_block(A, B, wmap, step):
    """(start, n, insert, r) of the single moved block that best explains the
    window map, or None if the map is the identity / not of that form."""
    T = len(A)
    off = wmap[:, 1] - wmap[:, 0]
    good = wmap[:, 2] > 0.8
    shifted = good & (off > 50)
    if not shifted.any():
        return None
    n0 = int(np.median(off[shifted]))
    # boundaries from the longest runs of consecutive windows with one offset
    # (single stray matches near a boundary are ignored)
    start0 = int(wmap[longest_run(good & (np.abs(off - n0) <= 3)), 0].min())
    # windows of the moved block: raw frames inside [start0, start0 + n0) found later in her movie
    block = good & (off < -50) & (wmap[:, 1] >= start0 - 50) & (wmap[:, 1] < start0 + n0 + 50)
    if not block.any():
        return None
    ins = wmap[:, 0] - (wmap[:, 1] - start0)
    insert0 = int(np.median(ins[block]))
    block &= np.abs(ins - insert0) <= 3
    insert0 = int(np.median(ins[longest_run(block)]))
    insert0 = min(max(insert0, start0), T - n0)
    # the window starts lie on a grid and windows straddling a boundary match
    # poorly: exhaustive search of start and insert over +-(step + 10) frames,
    # block length within +-3 of the median offset
    best = None
    for n in range(n0 - 3, n0 + 4):
        for start in range(max(start0 - step - 10, 1), start0 + step + 11):
            for insert in range(max(insert0 - step - 10, start), min(insert0 + step + 11, T - n) + 1):
                r = np.corrcoef(A[reorder(T, start, n, insert)], B)[0, 1]
                if best is None or r > best[3]:
                    best = (start, n, insert, r)
    return best


def xcorr(A, B, max_lag):
    lags = np.arange(-max_lag, max_lag + 1)
    T = len(A)
    r = [np.corrcoef(B[max(0, -L):T - max(0, L)], A[max(0, L):T + min(0, L)])[0, 1] for L in lags]
    return lags, np.array(r)


def main():
    p = parser(__doc__, "frame_order")
    p.add_argument("--window", type=int, default=100, help="window for the frame map, frames (default 100)")
    p.add_argument("--step", type=int, default=25, help="step between map windows, frames (default 25)")
    p.add_argument("--max-lag", type=int, default=700, help="constant-shift scan range, frames (default 700)")
    a = p.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root))
    out = Path(os.path.expanduser(a.out or default_out_dir(root, "frame_order")))
    out.mkdir(parents=True, exist_ok=True)
    sessions = [s for s in find_sessions(root)
                if not a.session or any(k in s.tag or k in s.label for k in a.session)]
    todo = []
    for s in sessions:
        ours = Path(root) / "raw_funcimg" / s.tag / "preprocessed" / "suite2p" / "plane0"
        if not (ours / "ops.npy").exists():
            print(f"{s.label}: no suite2p run on the raw tif in {ours}, skipped")
            continue
        todo.append((s, ours, Path(s.files["suite2p"])))
    if not todo:
        sys.exit("no session with our raw suite2p run")
    fig, axes = new_figure(len(todo), 2, figsize=(11, 2.8 * len(todo)), squeeze=False,
                           gridspec_kw={"width_ratios": [1, 1.3]})
    rows = []
    for (s, ours, hers), (ax_m, ax_x) in zip(todo, axes):
        a_, b_, kind = signals(ours, hers, out / f"{safe(s.label)}_signals.npy")
        T = len(b_)
        A, B = highpass(a_, 31), highpass(b_, 31)
        wmap = window_map(A, B, a.window, a.step)
        A9, B9 = highpass(a_, 9), highpass(b_, 9)
        fit = fit_block(A9, B9, wmap, a.step)
        lags, r = xcorr(A, B, a.max_lag)
        r_id = np.corrcoef(A9, B9)[0, 1]
        row = {"session": s.label, "signal": kind, "n_frames": T, "n_frames_ours": len(a_),
               "r_identity": r_id, "best_shift": int(lags[np.nanargmax(r)]), "r_best_shift": np.nanmax(r)}
        if fit:
            start, n, insert, rf = fit
            m = reorder(T, start, n, insert)
            np.save(out / f"{safe(s.label)}_her_to_raw.npy", m)
            row.update(block_start=start, block_frames=n, insert_at=insert, r_reordered=rf)
            print(f"{s.label} ({kind}): raw frames {start}-{start + n - 1} moved to her frame {insert}; "
                  f"r {r_id:.2f} as stored -> {rf:.2f} reordered")
        else:
            print(f"{s.label} ({kind}): no moved block found; r as stored {r_id:.2f}")
        rows.append(row)

        style_axis(ax_m)
        ax_m.plot([0, T], [0, T], color=GRIDLINE, linewidth=1, label="same order")
        if fit:
            ax_m.plot(np.arange(T), m, color=SERIES_2, linewidth=1, label="fitted reorder")
        ok = wmap[:, 2] > 0.8
        ax_m.scatter(wmap[ok, 0], wmap[ok, 1], s=6, color=SERIES_1, label="window match r > 0.8", zorder=3)
        ax_m.scatter(wmap[~ok, 0], wmap[~ok, 1], s=6, color=INK_MUTED, label="r ≤ 0.8", zorder=3)
        title(ax_m, f"{s.label}: Arya's frame → raw frame ({kind})")
        labels(ax_m, "Arya's frame", "raw frame (our suite2p)")
        ax_m.legend(fontsize=7, frameon=False, loc="upper left")

        style_axis(ax_x)
        ax_x.plot(lags, r, color=SERIES_1, linewidth=1)
        ax_x.axvline(0, color=GRIDLINE, linewidth=1)
        title(ax_x, f"Constant shift: best {row['best_shift']} frames (r {row['r_best_shift']:.2f}), "
                    f"r at 0 {r[lags == 0][0]:.2f}")
        labels(ax_x, "shift L (her frame k vs raw frame k + L)", "r (high-passed signals)")
    save(fig, out / "frame_order.png", "Frame order of Arya's processed movies vs our suite2p on the raw tif")
    pd.DataFrame(rows).to_csv(out / "frame_order.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
