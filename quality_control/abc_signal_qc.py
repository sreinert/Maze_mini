"""
Signal quality of the ABC-task extractions, with the same metrics and figures
as suite2p_signal_qc.py (so values compare with the fscan tests and with
Bonsai_ABCD), plus motion checks that only make sense in freely moving mice.

    uv run python quality_control/abc_signal_qc.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding
    uv run python quality_control/abc_signal_qc.py <root> --movie        # + frame-correlation motion proxy (slow)

Metrics (signal_quality_metrics.compute_percell_metrics, on F.npy / Fneu.npy of
suite2p_output/<tag>_...; all ROIs, iscell is all 1 there): F0, noise, SNR, skew,
event fraction, F-Fneu correlation, correlation with the population mean.

The fluorescence of these extractions has no meaningful zero: an offset was
removed upstream (bidi correction / denoising), the mean images have negative
pixels and in 4 of 6 AB_08 sessions most ROIs have an 8th-percentile F <= 0.
So traces are NOT divided by F0 here (normalise=False): 'dff' columns are dF in
F units. SNR, skew, event fraction and pop_corr are scale-free and compare with
other data; F0, noise_sd and peak_dff do not. The same holds for the
colleague's dff.npy / aligned activity (suite2p preprocess(Fcorr) = baseline-
subtracted Fcorr, not dF/F): z-score per cell before comparing amplitudes.
fs is the measured frame rate (~7.52 Hz, abc_data/sync.py), meanImg the masks/
chan1 mean tif, the structural channel the chan2 mean tif.

Motion checks (on the aligned window):
    speed coupling   correlation of the population-mean aligned activity with
                     head speed, against a circular-shift null; a shared
                     signal that follows running could be motion, not activity
    frame corr       (--movie) correlation of every --movie-step-th frame of
                     green_channel.bin with the session mean image: dips mark
                     residual motion / z-shifts after registration; also
                     correlated with speed and with the population mean

Outputs (in --out, default ~/Desktop/maze_mini_qc/<name of root>/signal_qc):
    traces/<session>.png   top-SNR and random cells (suite2p_signal_qc.plot_traces)
    summary.png            metric distributions per session (suite2p_signal_qc.plot_summary)
    motion/<session>.png   population mean, head speed and (--movie) frame
                           correlation over the session
    motion.png             speed coupling per session vs null
    summary.csv, per_cell.csv, motion.csv

Inputs are opened read-only; frames are read from the share, never cached there.
"""
import numpy as np
import pandas as pd

from abc_common import labels, new_figure, parser, safe, save, setup, title
from signal_quality_metrics import compute_channel2_metrics, compute_percell_metrics, summarise_traces
from suite2p_io import GRIDLINE, INK_MUTED, INK_PRIMARY, SERIES_1, SERIES_2, SERIES_3, style_axis
from suite2p_signal_qc import PER_CELL_COLS, plot_summary, plot_traces


def session_traces(s, neucoeff, baseline_percentile, event_sigma, normalise=False):
    ops = {"fs": s.fs_measured, "tau": s.metadata.get("tau"), "meanImg": s.mean_image(1),
           "meanImg_chan2": s.mean_image(2)}
    t = compute_percell_metrics(s.s2p("F"), s.s2p("Fneu"), s.s2p("iscell"), ops,
                                baseline_percentile, event_sigma, neucoeff, normalise)
    t["plane_dir"] = s.s2p_dir
    return t


def head_speed(s):
    xy = np.asarray(s.xy, dtype=float)
    sp = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1)] * s.fs_measured
    return sp


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else np.nan


def shift_null(a, b, n=500, min_shift_s=30, fs=7.5, seed=0):
    """Correlations of a with circularly shifted b (shifts >= min_shift_s)."""
    rng = np.random.default_rng(seed)
    lo = int(min_shift_s * fs)
    shifts = rng.integers(lo, len(a) - lo, n)
    return np.array([_corr(a, np.roll(b, k)) for k in shifts])


def frame_correlation(s, step, ref):
    """Correlation of every step-th frame of green_channel.bin with ref (aligned
    window only). Returns (aligned frame indices, r)."""
    Ly, Lx = ref.shape
    fb = Ly * Lx * 2
    idx = np.arange(0, s.n_frames, step)
    r = np.full(len(idx), np.nan)
    z_ref = (ref - ref.mean()) / ref.std()
    with open(s.movie_path, "rb") as f:
        for j, k in enumerate(idx):
            f.seek(int(k + s.aligned_offset) * fb)
            fr = np.fromfile(f, dtype=np.int16, count=Ly * Lx).astype(np.float32).reshape(Ly, Lx)
            if fr.size == Ly * Lx and fr.std() > 0:
                r[j] = float(np.mean((fr - fr.mean()) / fr.std() * z_ref))
    return idx, r


def plot_motion(s, pop, speed, fc, out_path):
    n = 3 if fc is not None else 2
    fig, axes = new_figure(n, 1, figsize=(16, 2.6 * n + 1), sharex=True)
    for ax in axes:
        style_axis(ax)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    t = s.time
    axes[0].plot(t, pop, color=SERIES_1, linewidth=0.8)
    labels(axes[0], y="mean activity")
    title(axes[0], "Population mean of aligned activity (ROICaT cells present)")
    axes[1].plot(t, speed, color=SERIES_2, linewidth=0.6)
    labels(axes[1], y="speed (px/s)")
    title(axes[1], "Head speed")
    if fc is not None:
        idx, r = fc
        axes[2].plot(t[idx], r, color=SERIES_3, linewidth=0.8)
        labels(axes[2], y="r with mean image")
        title(axes[2], "Frame-to-mean correlation (green_channel.bin)")
    for ax in axes:
        for a_, b_, c_, _ in s.trial_times:
            ax.axvline(t[min(a_, len(t) - 1)], color=INK_MUTED, linewidth=0.5, alpha=0.5)
    labels(axes[-1], x="time (s); grey lines = A_on")
    save(fig, out_path, f"{s.label}  |  motion checks")


def main():
    p = parser(__doc__, "signal_qc")
    p.add_argument("--n-cells", type=int, default=10)
    p.add_argument("--window", type=float, default=300, help="trace window in s (0 = all; default 300)")
    p.add_argument("--trace-neucoeff", type=float, default=0.7)
    p.add_argument("--neucoeff", type=float, default=0, help="summary metrics (default 0 = raw F)")
    p.add_argument("--baseline-percentile", type=float, default=8)
    p.add_argument("--event-sigma", type=float, default=3)
    p.add_argument("--snr-threshold", type=float, default=5)
    p.add_argument("--skew-threshold", type=float, default=1)
    p.add_argument("--movie", action="store_true", help="also compute the frame-correlation motion proxy")
    p.add_argument("--movie-step", type=int, default=4, help="use every n-th frame (default 4)")
    a = p.parse_args()
    root, out, sessions = setup(a, "signal_qc")
    for d in ("traces", "motion"):
        (out / d).mkdir(exist_ok=True)
    params = {"baseline_percentile": a.baseline_percentile, "event_sigma": a.event_sigma,
              "snr_threshold": a.snr_threshold, "skew_threshold": a.skew_threshold}

    rows, cells, motion = [], [], []
    for s in sessions:
        tr = session_traces(s, a.neucoeff, a.baseline_percentile, a.event_sigma)
        m = summarise_traces(tr, a.snr_threshold, a.skew_threshold)
        m.update(compute_channel2_metrics(s.s2p_dir, ops=tr["ops"]))
        rows.append({"plane": s.label, **m})
        c = pd.DataFrame({k: tr[k] for k in PER_CELL_COLS})
        c.insert(0, "roi", tr["cell_roi_idx"])
        c.insert(0, "plane", s.label)
        cells.append(c)
        plot_tr = tr if a.trace_neucoeff == a.neucoeff else session_traces(
            s, a.trace_neucoeff, a.baseline_percentile, a.event_sigma)
        note = (f"dF = Fc - F0c, Fc = F - {a.trace_neucoeff:g}*Fneu (a.u., not divided by F0)"
                if a.trace_neucoeff else "dF = F - F0 (a.u., not divided by F0)")
        plot_traces(plot_tr, s.label, out / "traces" / f"{safe(s.label)}.png", a.n_cells, a.window,
                    a.snr_threshold, a.trace_neucoeff, trace_note=note)

        act = np.asarray(s.activity)[:, s.present]
        pop = np.nanmean(act, axis=1)
        speed = head_speed(s)
        r_speed = _corr(pop, speed)
        null = shift_null(pop, speed, fs=s.fs_measured)
        mrow = {"session": s.label, "r_pop_speed": r_speed, "null_p95": float(np.nanpercentile(np.abs(null), 95)),
                "p_shift": float(np.mean(np.abs(null) >= abs(r_speed))),
                "median_cell_r_speed": float(np.nanmedian([_corr(act[:, i], speed) for i in range(act.shape[1])]))}
        fc = None
        if a.movie:
            print(f"  reading every {a.movie_step}th frame of {s.movie_path} ...", flush=True)
            fc = frame_correlation(s, a.movie_step, s.mean_image(1).astype(np.float32))
            idx, r = fc
            mrow.update({"frame_corr_median": float(np.nanmedian(r)),
                         "frame_corr_p1": float(np.nanpercentile(r, 1)),
                         "r_framecorr_speed": _corr(r, speed[idx]), "r_framecorr_pop": _corr(r, pop[idx])})
        motion.append(mrow)
        plot_motion(s, pop, speed, fc, out / "motion" / f"{safe(s.label)}.png")
        print(f"{s.label}: {m['n_cells']} ROIs, median SNR {m['median_snr']:.1f}, "
              f"{m['frac_cells_snr_gt_thresh']:.0%} SNR > {a.snr_threshold}, median r_pop {m['median_pop_corr']:.2f}, "
              f"r(pop, speed) {r_speed:.2f} (null 95% {mrow['null_p95']:.2f})")

    summary = pd.DataFrame(rows)
    summary["chan1_chan2_ratio"] = summary["mean_img_mean"] / summary["fov_chan2_mean"]
    per_cell = pd.concat(cells, ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    per_cell.to_csv(out / "per_cell.csv", index=False)
    plot_summary(summary, per_cell, out / "summary.png",
                 f"Signal quality ({'F - %g*Fneu' % a.neucoeff if a.neucoeff else 'raw F'})  |  {root}", params)
    motion = pd.DataFrame(motion)
    motion.to_csv(out / "motion.csv", index=False)

    fig, ax = new_figure(1, 1, figsize=(8, 5))
    style_axis(ax)
    x = np.arange(len(motion))
    ax.bar(x, motion.r_pop_speed, color=SERIES_1, edgecolor="white", label="r(population mean, speed)", zorder=2)
    ax.plot(x, motion.null_p95, "_", color=INK_PRIMARY, markersize=30, markeredgewidth=2,
            label="95th pct. |r| of shifted null", zorder=3)
    ax.plot(x, -motion.null_p95, "_", color=INK_PRIMARY, markersize=30, markeredgewidth=2, zorder=3)
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split("/")[-1] for l in motion.session], rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8)
    title(ax, "Population activity vs head speed")
    labels(ax, y="correlation")
    save(fig, out / "motion.png")
    print(f"Saved figures and tables to {out}")


if __name__ == "__main__":
    main()
