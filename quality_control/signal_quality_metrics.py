"""
Raw-signal quality metrics for suite2p output, computed directly from F.npy/Fneu.npy.

Ported from Bonsai_ABCD/preprocessing/quality_control/signal_quality_metrics.py so
the numbers are directly comparable to the ABCD (head-fixed) sessions. Metric
definitions are unchanged:

    F0          8th percentile of each cell's raw F (no neuropil subtraction)
    dF/F        (F - F0) / F0
    noise_sd    frame-to-frame-difference MAD of dF/F, scaled to an SD
    peak_dff    99th percentile of dF/F
    snr         peak_dff / noise_sd
    skew        skewness of dF/F
    event_frac  fraction of frames with dF/F > event_sigma * noise_sd
    neu_corr    correlation of raw F with its neuropil trace Fneu
    pop_corr    correlation of each cell's dF/F with the mean dF/F of all cells
                (added here, not in Bonsai_ABCD: a value near 1 for most cells
                means the traces are dominated by a shared signal, e.g.
                neuropil, brightness or motion, rather than single-cell activity)

They are computed from the raw traces rather than suite2p's own stat fields, so
they do not depend on the suite2p version or on whether ROIs came from
suite2p's functional detection or an imported Cellpose segmentation.

Differences from the Bonsai_ABCD version:
  - Inputs are opened read-only and nothing is cached next to the data
    (see CLAUDE.md: raw data is never written to).
  - A non-positive noise estimate gives NaN SNR instead of inf.
  - Optional neuropil subtraction (neucoeff > 0). The default 0 keeps the
    Bonsai_ABCD definition (raw F). With neucoeff > 0:
        Fc   = F - neucoeff * Fneu
        dF/F = (Fc - F0c) / F0
    where F0c is the baseline percentile of Fc and F0 that of raw F. The
    denominator is the raw baseline on purpose: in the Mini2P data the neuropil
    is about as bright as the cells, so F0c is often close to or below 0 and
    dividing by it blows dF/F up by orders of magnitude. With the raw baseline,
    dF/F stays "change as a fraction of the cell's own fluorescence"; per-cell
    SNR, skew and event fraction do not depend on the denominator.

Channel naming: suite2p saves the *functional* channel as F.npy / meanImg and
the other one as *_chan2, whatever `functional_chan` was. In the Mini2P data the
functional channel is acquired as channel 2 (functional_chan = 2), so `chan2`
here means the structural channel.
"""
import os
import warnings

import numpy as np
from scipy import stats

from suite2p_io import load_npy

REQUIRED_FILES = ("F.npy", "Fneu.npy", "iscell.npy", "ops.npy")


def _load_ops(plane_dir):
    return load_npy(os.path.join(plane_dir, "ops.npy"), allow_pickle=True).item()


def _rowwise_corr(a, b):
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(a * b, axis=1) / np.sqrt(np.nansum(a ** 2, axis=1) * np.nansum(b ** 2, axis=1))


def load_percell_traces(plane_dir, baseline_percentile=8, event_sigma=3, neucoeff=0.0, normalise=True,
                        all_rois=False):
    """Load raw traces and compute per-cell dF/F and quality metrics for one
    suite2p plane folder.

    Returns a dict with 'error' on failure, otherwise:
        cell_roi_idx : indices into the original (all-ROI) F.npy rows that
            are iscell==1 (or all ROIs, if none are marked as cells)
        F, dff : (n_cells, n_frames) raw F and dF/F (see module docstring for
            the neuropil-corrected definition)
        pop_mean : (n_frames,) mean dF/F over cells with a valid baseline
        F0 (baseline of raw F), noise_sd, peak_dff, skew, snr, event_frac, neu_corr, pop_corr : (n_cells,)
        fs, tau, n_frames, n_roi_total, ops : session-level values
    """
    try:
        missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(plane_dir, f))]
        if missing:
            raise FileNotFoundError(f"missing {missing}")
        F = np.asarray(load_npy(os.path.join(plane_dir, "F.npy")), dtype=np.float64)
        Fneu = np.asarray(load_npy(os.path.join(plane_dir, "Fneu.npy")), dtype=np.float64)
        iscell = np.asarray(load_npy(os.path.join(plane_dir, "iscell.npy")))
        if all_rois:   # every suite2p ROI counts as a cell (classifier label ignored)
            iscell = np.ones((F.shape[0], 2))
        ops = _load_ops(plane_dir)
    except Exception as e:
        return {"error": str(e), "plane_dir": plane_dir}
    out = compute_percell_metrics(F, Fneu, iscell, ops, baseline_percentile, event_sigma, neucoeff, normalise)
    out["plane_dir"] = plane_dir
    return out


def compute_percell_metrics(F, Fneu, iscell, ops, baseline_percentile=8, event_sigma=3, neucoeff=0.0,
                            normalise=True):
    """The metrics of load_percell_traces() from arrays: F, Fneu (n_roi, n_frames),
    iscell (n_roi, 2) and an ops-like dict (fs, tau, meanImg; only passed through).
    Used for extractions without a suite2p plane folder (e.g. the ABC data).

    normalise=False: 'dff' is F - F0 (or Fc - F0c) without dividing by F0, for
    data whose fluorescence has no meaningful zero (an offset was subtracted, so
    F0 can be <= 0). noise_sd and peak_dff are then in F units; snr, skew,
    event_frac and pop_corr do not depend on the division and stay comparable."""
    F = np.asarray(F, dtype=np.float64)
    Fneu = np.asarray(Fneu, dtype=np.float64)
    iscell = np.asarray(iscell)
    cell_mask = iscell[:, 0].astype(bool)
    if cell_mask.sum() == 0:
        cell_mask = np.ones(F.shape[0], dtype=bool)

    Fc = F[cell_mask]
    Fneu_c = Fneu[cell_mask]
    neu_corr = _rowwise_corr(Fc, Fneu_c)

    # cells whose raw baseline is <= 0 get all-NaN dF/F; silence the resulting
    # all-NaN-slice warnings, they show up as NaN metrics instead
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        F0 = np.percentile(Fc, baseline_percentile, axis=1, keepdims=True)
        denom = F0.copy()
        denom[denom <= 0] = np.nan
        if not normalise:
            denom = np.ones_like(F0)
        else:
            F0 = denom
        if neucoeff:
            Fcorr = Fc - neucoeff * Fneu_c
            F0c = np.percentile(Fcorr, baseline_percentile, axis=1, keepdims=True)
            dff = (Fcorr - F0c) / denom
        else:
            dff = (Fc - F0) / denom

        # frame-to-frame diff MAD: robust noise estimate insensitive to slow drift/transients
        diffs = np.diff(dff, axis=1)
        noise_sd = 1.4826 * np.median(np.abs(diffs - np.median(diffs, axis=1, keepdims=True)), axis=1) / np.sqrt(2)
        noise_sd[~(noise_sd > 0)] = np.nan

        peak_dff = np.nanpercentile(dff, 99, axis=1)
        skew = stats.skew(dff, axis=1, nan_policy="omit")
        skew = np.asarray(np.ma.filled(skew, np.nan), dtype=np.float64)
        snr = peak_dff / noise_sd
        event_frac = np.mean(dff > event_sigma * noise_sd[:, None], axis=1)
        event_frac[~np.isfinite(noise_sd)] = np.nan

        pop_mean = np.nanmean(dff, axis=0)
        pop_corr = _rowwise_corr(dff, np.broadcast_to(pop_mean, dff.shape))

    return {
        "normalised": normalise,
        "fs": ops.get("fs"),
        "tau": ops.get("tau"),
        "ops": ops,
        "n_frames": F.shape[1],
        "n_roi_total": F.shape[0],
        "cell_roi_idx": np.nonzero(cell_mask)[0],
        "F": Fc,
        "F0": F0[:, 0],
        "dff": dff,
        "noise_sd": noise_sd,
        "peak_dff": peak_dff,
        "skew": skew,
        "snr": snr,
        "event_frac": event_frac,
        "neu_corr": neu_corr,
        "pop_corr": pop_corr,
        "pop_mean": pop_mean,
    }


def summarise_traces(traces, snr_threshold=5, skew_threshold=1):
    """Plane-level summary metrics from a load_percell_traces() dict."""
    mean_img = traces["ops"].get("meanImg")
    mean_img_stats = {}
    if mean_img is not None:
        mean_img_stats = {
            "mean_img_mean": float(np.mean(mean_img)),
            "mean_img_median": float(np.median(mean_img)),
            "mean_img_p99": float(np.percentile(mean_img, 99)),
        }

    return {
        "plane_dir": traces["plane_dir"],
        "fs": traces["fs"],
        "tau": traces["tau"],
        "n_frames": traces["n_frames"],
        "n_cells": len(traces["cell_roi_idx"]),
        "n_roi_total": traces["n_roi_total"],
        "baseline_F_median": float(np.nanmedian(traces["F0"])),
        "median_skew_dff": float(np.nanmedian(traces["skew"])),
        "median_peak_dff_pct99": float(np.nanmedian(traces["peak_dff"])),
        "median_snr": float(np.nanmedian(traces["snr"])),
        "frac_cells_snr_gt_thresh": float(np.mean(traces["snr"] > snr_threshold)),
        "frac_cells_skew_gt_thresh": float(np.mean(traces["skew"] > skew_threshold)),
        "median_event_frac": float(np.nanmedian(traces["event_frac"])),
        "median_neu_corr": float(np.nanmedian(traces["neu_corr"])),
        "median_pop_corr": float(np.nanmedian(traces["pop_corr"])),
        **mean_img_stats,
    }


def compute_channel2_metrics(plane_dir, ops=None):
    """Brightness of the structural (activity-independent) channel, as a control
    for whether a dim functional channel reflects a general imaging problem (dim
    in both channels) or an indicator/expression-specific one (dim only in the
    functional channel).

    ROI brightness applies each ROI's lambda-weighted mask (stat.npy) to
    ops['meanImg_chan2'], so it works whether or not F_chan2.npy was extracted.
    Where F_chan2.npy exists, its time-median and skew are also reported
    (skew should stay near 0 for a structural channel).
    """
    try:
        if ops is None:
            ops = _load_ops(plane_dir)
        stat = load_npy(os.path.join(plane_dir, "stat.npy"), allow_pickle=True)
        iscell = np.asarray(load_npy(os.path.join(plane_dir, "iscell.npy")))
    except Exception as e:
        return {"error": str(e)}

    mean_img2 = ops.get("meanImg_chan2")
    if mean_img2 is None:
        return {"error": "no meanImg_chan2 in ops"}

    cell_mask = iscell[:, 0].astype(bool)
    if cell_mask.sum() == 0:
        cell_mask = np.ones(len(stat), dtype=bool)

    roi_vals = []
    for s in stat[cell_mask]:
        lam = s["lam"] / s["lam"].sum()
        roi_vals.append(float(np.sum(lam * mean_img2[s["ypix"], s["xpix"]])))
    roi_vals = np.array(roi_vals)

    result = {
        "fov_chan2_mean": float(np.mean(mean_img2)),
        "fov_chan2_median": float(np.median(mean_img2)),
        "fov_chan2_p99": float(np.percentile(mean_img2, 99)),
        "roi_chan2_median_brightness": float(np.median(roi_vals)),
    }

    f2_path = os.path.join(plane_dir, "F_chan2.npy")
    if os.path.exists(f2_path):
        F2 = np.asarray(load_npy(f2_path), dtype=np.float64)[cell_mask]
        result["roi_chan2_F_time_median"] = float(np.median(np.median(F2, axis=1)))
        result["roi_chan2_F_time_skew_median"] = float(np.nanmedian(stats.skew(F2, axis=1)))
    return result


def select_cells(traces, n_cells=10, order="snr", seed=0):
    """Indices (into the iscell-filtered arrays) of cells to plot.

    order='snr'    : top n_cells by SNR (NaN SNR pushed to the back)
    order='random' : n_cells drawn at random from cells with finite SNR, so the
                     plot shows what a typical cell looks like, not just the best
    """
    snr = traces["snr"]
    finite = np.flatnonzero(np.isfinite(snr))
    if len(finite) == 0:
        return np.array([], dtype=int)
    n = min(n_cells, len(finite))
    if order == "random":
        return np.sort(np.random.default_rng(seed).choice(finite, size=n, replace=False))
    return np.argsort(np.where(np.isfinite(snr), snr, -np.inf))[::-1][:n]
