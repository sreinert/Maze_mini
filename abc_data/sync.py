"""
Clocks and synchronisation.

Three clocks, tied together by the pyControl rsync pulses:

    imaging DAQ   h5 (5 kHz): FrameClock (one rising edge per imaging frame) and
                  PyControlPulse (copy of the pyControl rsync line)
    pyControl     .tsv times; 'rsync' events
    camera        _GPIO_data.csv: one row per video frame, Timestamp in ns,
                  GPIO1 = rsync line

FrameClock is high at sample 0 in most files, followed by ~3 s of silence
before the first real frame edge: a DAQ start-up artefact, not a frame. Only
low-to-high transitions are counted. In some sessions that gives one edge fewer
than suite2p has frames; the missing frame is taken to be the last one (the DAQ
stopped before its edge) and its time is extrapolated by one median interval
(load_imaging_sync(..., n_frames=...)). That is the convention under which the
colleague's aligned arrays match suite2p (abc_sync_qc).

The real frame interval is ~0.1330 s (~7.52 Hz), not the nominal fs = 7.5 in
extraction_metadata: converting frame counts to time with 7.5 drifts by ~0.25 %
(~2 s by the end of a 13 min session). Use frame_times for timing.
"""
from dataclasses import dataclass

import numpy as np


def rising_edges(x, threshold=None):
    """Sample indices of low-to-high transitions (threshold = mid-range)."""
    x = np.asarray(x, dtype=np.float64)
    if threshold is None:
        threshold = (np.nanmax(x) + np.nanmin(x)) / 2
    high = x > threshold
    return np.flatnonzero(~high[:-1] & high[1:]) + 1, bool(high[0])


@dataclass
class ClockFit:
    """dst = slope * src + intercept, fitted on matched sync pulses."""
    slope: float
    intercept: float
    residuals: np.ndarray  # dst - fit, s
    n: int

    def __call__(self, t):
        return self.slope * np.asarray(t, dtype=np.float64) + self.intercept

    @property
    def max_abs_residual(self):
        return float(np.max(np.abs(self.residuals))) if self.n else np.nan


def fit_clock(src, dst):
    """Linear fit between two sets of pulse times. They must have the same count
    (the pulses are matched one to one); otherwise raises."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError(f"cannot match pulses: {len(src)} vs {len(dst)}")
    slope, intercept = np.polyfit(src, dst, 1)
    return ClockFit(slope, intercept, dst - (slope * src + intercept), len(src))


@dataclass
class ImagingSync:
    samplerate: float
    frame_times: np.ndarray       # s, DAQ clock; start of each imaging frame
    pulse_times: np.ndarray       # s, DAQ clock; pyControl rsync
    n_edges: int                  # FrameClock rising edges found
    starts_high: bool
    duration: float
    n_extrapolated: int = 0       # frames appended at the end (see module doc)

    @property
    def frame_intervals(self):
        return np.diff(self.frame_times)


def load_imaging_sync(h5_path, n_frames=None):
    """Frame and pulse times from an imaging DAQ h5. With n_frames (suite2p frame
    count), up to 2 missing trailing frames are extrapolated (see module doc)."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        sr = float(f.attrs["samplerate"])
        fc = f["FrameClock"][:]
        pc = f["PyControlPulse"][:]
    fe, fc_high = rising_edges(fc)
    pe, _ = rising_edges(pc)
    ft = fe / sr
    n_extra = 0
    if n_frames is not None and 0 < n_frames - len(ft) <= 2:
        n_extra = n_frames - len(ft)
        ft = np.r_[ft, ft[-1] + np.median(np.diff(ft)) * np.arange(1, n_extra + 1)]
    return ImagingSync(sr, ft, pe / sr, len(fe), fc_high, len(fc) / sr, n_extra)


@dataclass
class CameraSync:
    frame_times: np.ndarray       # s from the first video frame
    pulse_times: np.ndarray       # s, GPIO1 rising edges
    gpio: np.ndarray              # (n_frames, 3)

    @property
    def frame_intervals(self):
        return np.diff(self.frame_times)


def load_camera_sync(gpio_csv, sync_line=1):
    data = np.loadtxt(gpio_csv, delimiter=",", skiprows=1, dtype=np.int64)
    t = (data[:, 3] - data[0, 3]) / 1e9
    g = data[:, :3]
    x = g[:, sync_line - 1]
    rises = np.flatnonzero((x[1:] > 0) & (x[:-1] == 0)) + 1
    return CameraSync(t, t[rises], g)


def time_to_frame(t, frame_times):
    """Index of the frame that contains each time (frame i spans
    [frame_times[i], frame_times[i+1])); -1 before the first frame."""
    return np.searchsorted(frame_times, np.asarray(t, float), side="right") - 1
