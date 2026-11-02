"""
Sessions processed by us from the raw ABC acquisitions (suite2p on the raw
ScanImage tif), synced to behaviour from the raw sync signals.

Layout (e.g. /Volumes/behrens/Sandra/mini2p/Arya/RAW_GOOD):

    funcimg/<session>/<animal>_<n>_<nnnnn>_<date>_<time>.tif   raw ScanImage stack
    funcimg/<session>/file_<id>_<date>_<time>.h5               imaging DAQ
    funcimg/<session>/preprocessed/suite2p/plane0/             suite2p output
    behaviour/<animal>-<date>-<HHMMSS>.tsv                     pyControl log (matched by date)
    video/<animal>_<date>-<HHMMSS>.mp4 (+ _GPIO_data.csv)      camera (matched by date)

The same layout with raw_funcimg / raw_behaviour / raw_video (our suite2p runs
next to the raw tifs inside Arya's processed folder, e.g. PROCESSED/good_decoding)
is found too; planes without F.npy (registration only) are skipped.

Time base: suite2p frame i starts at DAQ FrameClock edge i (a missing trailing
edge is extrapolated, see sync.py); pyControl events map to frames through the
rsync clock fit. Nothing from Arya's aligned arrays is used.

    from abc_data.raw import find_raw_sessions
    s = find_raw_sessions(root)[0]
    s.goal_progress(10)     # (n_frames,) bin per suite2p frame, -1 outside
    s.activity()            # (n_frames, n_cells) F - 0.7 Fneu of all suite2p ROIs (s.cells)
"""
import glob
import os
from functools import cached_property

import numpy as np

from .pycontrol import load_pycontrol, trial_events
from .sessions import TAG_RE
from types import SimpleNamespace

from .sync import fit_clock, load_camera_sync, load_imaging_sync, time_to_frame


class RawSession:
    def __init__(self, plane_dir, tif, h5, tsv, video=None, gpio=None):
        self.plane_dir, self.tif, self.h5, self.tsv = plane_dir, tif, h5, tsv
        self.video_path, self.gpio = video, gpio
        # same access pattern as sessions.SessionFiles for scripts shared with Arya's layout
        self.files = SimpleNamespace(files={"h5": h5, "tsv": tsv, "video": video, "gpio": gpio, "tif": tif})
        m = TAG_RE.search(os.path.basename(tif))
        self.tag = m.group(0)
        self.label = f"{m.group('animal')}/{m.group('date')}"
        self.ops = np.load(os.path.join(plane_dir, "ops.npy"), allow_pickle=True).item()

    def _npy(self, name):
        return np.load(os.path.join(self.plane_dir, f"{name}.npy"), mmap_mode="r")

    @property
    def n_frames(self):
        return self._npy("F").shape[1]

    @cached_property
    def iscell(self):
        """suite2p's classifier label (not used to select ROIs; see cells)."""
        return np.asarray(self._npy("iscell"))[:, 0].astype(bool)

    @cached_property
    def cells(self):
        """(n_rois,) bool, the ROIs analysed: all suite2p ROIs (the suite2p
        classifier's iscell is not used as a filter)."""
        return np.ones(self._npy("F").shape[0], bool)

    def activity(self, neucoeff=0.7, chan2=False):
        """(n_frames, n_cells) F - neucoeff * Fneu of the analysed ROIs (cells; all
        ROIs) of the functional channel (chan2=True: the other channel, as a control)."""
        sfx = "_chan2" if chan2 else ""
        F = np.asarray(self._npy("F" + sfx), dtype=np.float64)[self.cells]
        N = np.asarray(self._npy("Fneu" + sfx), dtype=np.float64)[self.cells]
        return (F - neucoeff * N).T

    @cached_property
    def pycontrol(self):
        return load_pycontrol(self.tsv)

    @cached_property
    def imaging_sync(self):
        return load_imaging_sync(self.h5, n_frames=self.n_frames)

    @cached_property
    def pyc_to_daq(self):
        return fit_clock(self.pycontrol.rsync_times(), self.imaging_sync.pulse_times)

    @property
    def task(self):
        """(3,) A, B, C node ids (pyControl active_poke)."""
        return np.asarray(self.pycontrol.active_poke, dtype=int)

    def goal_frames(self):
        """[A_on, B_on, C_on] suite2p frames (each an int array)."""
        out = []
        for n in ("A_on", "B_on", "C_on"):
            k = self.pyc_to_frame(self.pycontrol.state_times(n))
            out.append(k[np.isfinite(k)].astype(int))
        return out

    @property
    def n_s2p_frames(self):
        return self.n_frames

    @cached_property
    def camera_sync(self):
        return load_camera_sync(self.gpio)

    @cached_property
    def pyc_to_camera(self):
        return fit_clock(self.pycontrol.rsync_times(), self.camera_sync.pulse_times)

    @property
    def aligned_offset(self):
        """suite2p frame nearest pyControl t = 0 (no aligned arrays here; for plots
        shared with Arya's layout)."""
        return int(self.pyc_to_frame(0.0))

    def pyc_to_camera_frame(self, t):
        return time_to_frame(self.pyc_to_camera(t), self.camera_sync.frame_times)

    def mean_image(self, chan=1):
        """Registered mean image of suite2p's functional channel (chan=1) or the
        other one (chan=2)."""
        return self.ops["meanImg" if chan == 1 else "meanImg_chan2"]

    @property
    def fs_measured(self):
        return 1.0 / float(np.median(self.imaging_sync.frame_intervals))

    def pyc_to_frame(self, t):
        """pyControl times -> nearest suite2p frame (NaN outside the recording)."""
        ft = self.imaging_sync.frame_times
        pos = np.interp(self.pyc_to_daq(t), ft, np.arange(len(ft)), left=np.nan, right=np.nan)
        return np.round(pos)

    def reward_frames(self):
        t = np.sort(np.concatenate([self.pycontrol.state_times(n) for n in ("A_on", "B_on", "C_on")]))
        k = self.pyc_to_frame(t)
        return k[np.isfinite(k)].astype(int)

    def goal_progress(self, n_bins=10):
        """As abc_data.loaders.Session.goal_progress, on suite2p frames."""
        gp = np.full(self.n_frames, -1)
        r = self.reward_frames()
        for a, b in zip(r[:-1], r[1:]):
            if b > a:
                gp[a:b] = np.minimum((np.arange(b - a) * n_bins) // (b - a), n_bins - 1)
        return gp

    def goal_segments(self):
        """(n_frames,) segment type of each frame: 0 = A->B, 1 = B->C, 2 = C->A (from
        the goal that starts the segment between two rewards), -1 outside; and
        (n_frames,) running segment number (-1 outside)."""
        names = ("A_on", "B_on", "C_on")
        t = np.concatenate([self.pycontrol.state_times(n) for n in names])
        g = np.concatenate([np.full(len(self.pycontrol.state_times(n)), i) for i, n in enumerate(names)])
        o = np.argsort(t)
        k = self.pyc_to_frame(t[o])
        ok = np.isfinite(k)
        k, g = k[ok].astype(int), g[o][ok]
        seg = np.full(self.n_frames, -1)
        num = np.full(self.n_frames, -1)
        for i, (a, b) in enumerate(zip(k[:-1], k[1:])):
            seg[a:b], num[a:b] = g[i], i
        return seg, num

    def global_signal(self, kind="background", neucoeff=0.7):
        """(n_frames,) global regressor: 'background' = mean Fneu over all ROIs
        (neuropil / whole-frame fluorescence), 'pop' = mean of activity() over cells."""
        if kind == "background":
            return np.asarray(self._npy("Fneu"), dtype=np.float64).mean(axis=0)
        if kind == "pop":
            return self.activity(neucoeff).mean(axis=1)
        raise ValueError(kind)

    def trial_index(self):
        """(n_frames,) trial number (A_on to next A_on; the last trial runs to the
        last reward), -1 outside."""
        ev = trial_events(self.pycontrol)
        idx = np.full(self.n_frames, -1)
        last = self.reward_frames()[-1] if len(self.reward_frames()) else 0
        for i, (a, _, _, nxt) in enumerate(ev):
            fa = self.pyc_to_frame(a)
            fe = self.pyc_to_frame(nxt) if np.isfinite(nxt) else last
            if np.isfinite(fa) and np.isfinite(fe):
                idx[int(fa):int(fe)] = i
        return idx

    @property
    def n_trials(self):
        return len(trial_events(self.pycontrol))


def find_raw_sessions(root):
    """RawSessions below root: every suite2p plane0 (with F.npy) under
    [raw_]funcimg/<session>/, with the tif and h5 of that folder and the
    pyControl log of the same date."""
    root = os.path.abspath(os.path.expanduser(root))
    pre = "raw_" if os.path.isdir(os.path.join(root, "raw_funcimg")) else ""
    out = []
    for plane in sorted(glob.glob(os.path.join(root, pre + "funcimg", "*", "preprocessed", "suite2p", "plane0"))):
        if not os.path.exists(os.path.join(plane, "F.npy")):
            print(f"{plane}: no F.npy (registration only), skipped")
            continue
        sess = plane.split(os.sep + "preprocessed")[0]
        tifs = [t for t in glob.glob(os.path.join(sess, "*.tif")) if TAG_RE.search(os.path.basename(t))]
        h5s = glob.glob(os.path.join(sess, "file_*.h5"))
        if len(tifs) != 1 or len(h5s) != 1:
            raise RuntimeError(f"{sess}: need exactly one tif and one h5, found {tifs} {h5s}")
        m = TAG_RE.search(os.path.basename(tifs[0]))
        tsv = glob.glob(os.path.join(root, pre + "behaviour", f"{m.group('animal')}-{m.group('date')}-*.tsv"))
        if len(tsv) != 1:
            raise RuntimeError(f"{sess}: {len(tsv)} pyControl logs for {m.group('date')}")
        vid = glob.glob(os.path.join(root, pre + "video", f"{m.group('animal')}_{m.group('date')}-??????.mp4"))
        gpio = glob.glob(os.path.join(root, pre + "video", f"{m.group('animal')}_{m.group('date')}-*_GPIO_data.csv"))
        out.append(RawSession(plane, tifs[0], h5s[0], tsv[0], vid[0] if len(vid) == 1 else None,
                              gpio[0] if len(gpio) == 1 else None))
    return out


class ProcessedSession(RawSession):
    """Arya's processed session (sessions.SessionFiles) seen through the RawSession
    interface: suite2p F / Fneu / iscell of suite2p_output/<tag>_..., frames timed by
    the DAQ FrameClock and behaviour from the pyControl log via the rsync fit,
    exactly as for our raw sessions. The aligned arrays (activity, trial_times,
    XY) are not used, so the colleague's alignment does not enter."""

    def __init__(self, files):
        f = files.files
        self.plane_dir, self.tif, self.h5, self.tsv = f["suite2p"], None, f["h5"], f["tsv"]
        self.video_path, self.gpio = f["video"], f["gpio"]
        self.files = SimpleNamespace(files=dict(f))
        self.tag, self.label = files.tag, files.label
        self.session_files = files   # to load the aligned arrays (e.g. maze_location) when needed
        meta = np.load(os.path.join(self.plane_dir, "extraction_metadata.npy"), allow_pickle=True).item()
        self.ops = {"fs": float(meta["fs"])}

    def mean_image(self, chan=1):
        import tifffile
        return tifffile.imread(self.files.files[f"mean_chan{chan}"])

    def activity(self, neucoeff=0.7, chan2=False):
        if chan2:
            raise ValueError("no second-channel traces in the processed extraction")
        return super().activity(neucoeff)


def find_processed_sessions(root):
    """ProcessedSession for every session of Arya's layout below root."""
    from .sessions import find_sessions
    return [ProcessedSession(f) for f in find_sessions(root)]
