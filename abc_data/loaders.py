"""
Load one ABC-task session (Arya's preprocessed layout, see sessions.py).

    from abc_data import find_sessions, load_session
    sessions = find_sessions("/Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding")
    s = load_session(sessions[0])
    s.activity          # (T, n_roicat) aligned activity, NaN columns = cell absent
    s.present           # (n_roicat,) bool
    s.xy, s.location    # (T, 2) video px, (T,) location id (maze.py)
    s.trial_times       # (n_trials, 4) aligned frames of A, B, C, next A
    s.task              # (3,) A, B, C node ids
    s.fs                # nominal frame rate (extraction_metadata; 7.5)
    s.time              # (T,) pyControl s of each aligned frame (real frame clock)

Everything in aligned_arrays shares one time base: imaging frames, starting at
the imaging frame nearest pyControl t = 0. Aligned frame k is
suite2p frame k + s.aligned_offset; s.aligned_offset is derived from the
DAQ/pyControl sync (not from the aligned arrays). s.suite2p_dff_aligned() gives
suite2p dff cut to the aligned window by that offset, independent of the
colleague's alignment, and s.roi_match() relates the ROICaT columns to suite2p
ROIs (abc_sync_qc / abc_roi_qc check both).

All arrays are opened read-only (memory-mapped where possible).
"""
import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from .pycontrol import load_pycontrol, trial_events
from .sessions import SessionFiles
from .sync import fit_clock, load_camera_sync, load_imaging_sync, time_to_frame


def _npy(path, allow_pickle=False):
    if allow_pickle:
        return np.load(path, allow_pickle=True)
    return np.load(path, mmap_mode="r")


def zscore_cols(x):
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)


def corr_matrix(a, b):
    """Pearson correlation between the columns of a (T, n) and b (T, m)."""
    za, zb = zscore_cols(a), zscore_cols(b)
    return np.nan_to_num(za).T @ np.nan_to_num(zb) / a.shape[0]


@dataclass
class RoiMatch:
    """ROICaT column <-> suite2p ROI correspondence for one session.

    corr       (n_roi, n_present) correlation of suite2p dff (aligned window) with
               each present ROICaT column
    columns    (n_present,) ROICaT column index of each corr column
    best_col   (n_roi,) best-matching ROICaT column per ROI
    best_r     (n_roi,) its correlation
    mutual     (n_roi,) the ROI is also the best ROI for that column
    """
    corr: np.ndarray
    columns: np.ndarray
    best_col: np.ndarray
    best_r: np.ndarray
    mutual: np.ndarray

    def matched(self, r_min=0.8):
        return self.mutual & (self.best_r >= r_min)


class Session:
    def __init__(self, files: SessionFiles):
        self.files = files
        self.tag, self.label = files.tag, files.label
        f = files.files
        self.s2p_dir = f["suite2p"]
        self.metadata = _npy(os.path.join(self.s2p_dir, "extraction_metadata.npy"), True).item()
        self.fs = float(self.metadata["fs"])

    # ---- aligned arrays (colleague's alignment) ----
    @cached_property
    def activity(self):
        return _npy(self.files.files["activity"])

    @cached_property
    def present(self):
        return ~np.all(np.isnan(self.activity), axis=0)

    @property
    def n_frames(self):
        return self.activity.shape[0]

    @cached_property
    def xy(self):
        return _npy(self.files.files["xy"])

    @cached_property
    def location(self):
        return np.asarray(_npy(self.files.files["location"])).astype(int)

    @cached_property
    def trial_times(self):
        return np.asarray(_npy(self.files.files["trial_times"])).astype(int)

    @cached_property
    def task(self):
        return np.asarray(_npy(self.files.files["task"])).astype(int)

    @property
    def time(self):
        """(T,) pyControl time (s) of the start of each aligned frame, from the DAQ
        frame clock (the real frame rate is ~7.52 Hz, not the nominal fs = 7.5)."""
        ft = self.imaging_sync.frame_times[self.aligned_offset:self.aligned_offset + self.n_frames]
        return (ft - self.pyc_to_daq.intercept) / self.pyc_to_daq.slope

    @property
    def fs_measured(self):
        return 1.0 / float(np.median(self.imaging_sync.frame_intervals))

    def phase(self):
        """(T,) task phase per aligned frame: 0 = A->B, 1 = B->C, 2 = C->A (the last
        trial's C->A runs to the end of the window), -1 = none."""
        ph = np.full(self.n_frames, -1)
        for a, b, c, nxt in self.trial_times:
            ph[a:b], ph[b:c], ph[c:nxt] = 0, 1, 2
        return ph

    def reward_frames(self):
        """Sorted aligned frames of all goal pokes (A_on, B_on, C_on = rewards), from
        pyControl through the DAQ sync; unlike trial_times this keeps the real time
        of the first A_on."""
        t = np.concatenate([self.pycontrol.state_times(n) for n in ("A_on", "B_on", "C_on")])
        k = self.pyc_to_aligned_frame(np.sort(t))
        return k[np.isfinite(k) & (k >= 0) & (k < self.n_frames)].astype(int)

    def goal_progress(self, n_bins=10):
        """(T,) goal-progress bin 0..n_bins-1 per aligned frame: the time from one
        reward to the next split into n_bins equal parts (bin 0 just after a
        reward, n_bins-1 just before the next). -1 before the first and after the
        last reward."""
        gp = np.full(self.n_frames, -1)
        r = self.reward_frames()
        for a, b in zip(r[:-1], r[1:]):
            if b > a:
                gp[a:b] = np.minimum((np.arange(b - a) * n_bins) // (b - a), n_bins - 1)
        return gp

    @property
    def n_trials_complete(self):
        """Trials followed by another A_on (the last trial usually runs to the end)."""
        return int(np.sum(~np.isnan(self.trial_events_pycontrol()[:, 3])))

    # ---- suite2p extraction ----
    def s2p(self, name, allow_pickle=False):
        return _npy(os.path.join(self.s2p_dir, f"{name}.npy"), allow_pickle)

    @cached_property
    def stat(self):
        return self.s2p("stat", True)

    @cached_property
    def roi_table(self):
        return pd.read_csv(os.path.join(self.s2p_dir, "roi_table.csv"))

    @property
    def n_s2p_frames(self):
        return self.s2p("F").shape[1]

    def mean_image(self, chan=1):
        import tifffile
        return tifffile.imread(self.files.files[f"mean_chan{chan}"])

    @property
    def movie_path(self):
        return os.path.join(self.s2p_dir, "green_channel.bin")

    # ---- behaviour, video, sync ----
    @cached_property
    def pycontrol(self):
        return load_pycontrol(self.files.files["tsv"])

    @cached_property
    def imaging_sync(self):
        return load_imaging_sync(self.files.files["h5"], n_frames=self.n_s2p_frames)

    @cached_property
    def camera_sync(self):
        return load_camera_sync(self.files.files["gpio"])

    @property
    def video_path(self):
        return self.files.files["video"]

    @cached_property
    def pyc_to_daq(self):
        """ClockFit: pyControl s -> imaging DAQ s."""
        return fit_clock(self.pycontrol.rsync_times(), self.imaging_sync.pulse_times)

    @cached_property
    def pyc_to_camera(self):
        """ClockFit: pyControl s -> camera s (from the first video frame)."""
        return fit_clock(self.pycontrol.rsync_times(), self.camera_sync.pulse_times)

    def pyc_to_frame_position(self, t):
        """pyControl times -> continuous suite2p frame position (frame i starts at i)."""
        ft = self.imaging_sync.frame_times
        return np.interp(self.pyc_to_daq(t), ft, np.arange(len(ft)), left=np.nan, right=np.nan)

    @property
    def offset_at_pyc_start(self):
        """Nearest suite2p frame to pyControl t = 0 (from the sync alone)."""
        return int(np.round(self.pyc_to_frame_position(0.0)))

    @cached_property
    def offset_votes(self):
        """round(event frame position) - given trial_times for every A/B/C_on event
        (the first trial's A, set to 0 in trial_times, is left out)."""
        ev = self.trial_events_pycontrol()
        given = self.trial_times.astype(float)[:len(ev)]
        ok = np.isfinite(ev)
        ok[0, 0] = False
        return (np.round(self.pyc_to_frame_position(ev[ok])) - given[ok]).astype(int)

    @cached_property
    def aligned_offset(self):
        """suite2p frame index of aligned frame 0.

        Taken as the most common offset between the given trial_times and the
        pyControl events mapped to frames through the DAQ sync (nearest frame),
        so it uses no neural data; roi_match() then tests whether the aligned
        activity sits at the same offset. It is the nearest frame to pyControl
        t = 0 (offset_at_pyc_start) in 5 of the 6 AB_08 sessions and one frame
        later in 08-29 (t = 0 falls 0.45 frame into a frame there)."""
        v = self.offset_votes
        return int(np.bincount(v - v.min()).argmax() + v.min()) if len(v) else self.offset_at_pyc_start

    def pyc_to_aligned_frame(self, t):
        """pyControl times -> nearest aligned frame index (via the DAQ sync)."""
        return np.round(self.pyc_to_frame_position(t)) - self.aligned_offset

    def aligned_frame_to_camera_frame(self, k):
        """Aligned frame index -> video frame index at the start of that imaging frame."""
        return time_to_frame(self.pyc_to_camera(self.time[np.asarray(k)]), self.camera_sync.frame_times)

    def trial_events_pycontrol(self):
        """(n_trials, 4) pyControl times of A_on, B_on, C_on, next A_on."""
        return trial_events(self.pycontrol)

    def trial_times_resynced(self):
        """trial_times recomputed from pyControl + DAQ sync, in aligned frames (the
        next-A of the last trial is NaN when there is none)."""
        ev = self.trial_events_pycontrol()
        out = np.full(ev.shape, np.nan)
        ok = np.isfinite(ev)
        out[ok] = self.pyc_to_aligned_frame(ev[ok])
        return out

    # ---- neural data, two routes ----
    def suite2p_dff_aligned(self, name="dff", offset=None):
        """(T, n_roi) suite2p trace cut to the aligned window via the sync offset
        (NaN where the window runs past the recording)."""
        off = self.aligned_offset if offset is None else offset
        y = np.asarray(self.s2p(name), dtype=np.float32)
        out = np.full((self.n_frames, y.shape[0]), np.nan, dtype=np.float32)
        lo, hi = max(off, 0), min(off + self.n_frames, y.shape[1])
        if hi > lo:
            out[lo - off:hi - off] = y[:, lo:hi].T
        return out

    def roi_match(self, offset=None):
        cols = np.flatnonzero(self.present)
        dff = self.suite2p_dff_aligned(offset=offset)
        ok = np.all(np.isfinite(dff), axis=1)
        c = corr_matrix(dff[ok], np.asarray(self.activity)[ok][:, cols])
        best = c.argmax(axis=1)
        best_roi_per_col = c.argmax(axis=0)
        mutual = best_roi_per_col[best] == np.arange(c.shape[0])
        return RoiMatch(c, cols, cols[best], c[np.arange(c.shape[0]), best], mutual)


def load_session(files: SessionFiles):
    return Session(files)
