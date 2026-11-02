"""
Find the sessions of a preprocessed ABC-task folder and the files that belong to
each one.

Layout (Arya's preprocessing, e.g. /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding):

    aligned_arrays/neuronal_activity/<animal>_roicat_activity_session_<k>_<tag>_bidicorrected_denoised_registered.npy
    aligned_arrays/XY_position/XY_raw_<tag>.npy
    aligned_arrays/maze_location/Location_raw_<tag>.npy
    aligned_arrays/trial_times/trialtimes_<tag>.npy
    aligned_arrays/tasks_persess/Task_data_<tag>.npy
    suite2p_output/<tag>_bidicorrected_denoised_registered/{F,Fneu,dff,stat,...}.npy
    masks/<tag>_bidicorrected_denoised_registered_mean_{chan1,chan2,composite}.tif (+ _seg.npy)
    processed_funcimg/file_<id>_<date>_<time>.h5        imaging DAQ: FrameClock, PyControlPulse
    raw_behaviour/<animal>-<date>-<HHMMSS>.tsv         pyControl log
    raw_video/<animal>_<date>-<HHMMSS>.mp4 (+ _GPIO_data.csv)

<tag> is the imaging name <animal>_<n>_<nnnnn>_<YYYY-MM-DD>_<HH-MM-SS>; the h5 file
shares its date and time. Behaviour and video file names carry a different
(local-time) stamp, so they are matched to the imaging session by date, which
must be unique.
"""
import glob
import os
import re
from dataclasses import dataclass, field

TAG_RE = re.compile(r"(?P<animal>[A-Za-z]+_\d+)_(?P<n>\d+)_(?P<run>\d+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})")
S2P_SUFFIX = "_bidicorrected_denoised_registered"


@dataclass
class SessionFiles:
    tag: str
    animal: str
    date: str
    time: str
    roicat_session: int | None = None
    files: dict = field(default_factory=dict)

    @property
    def label(self):
        """Short label, e.g. AB_08/2026-08-13."""
        return f"{self.animal}/{self.date}"

    def missing(self):
        return [k for k, v in self.files.items() if v is None]


def _one(pattern, what, tag):
    hits = sorted(glob.glob(pattern))
    if len(hits) > 1:
        raise RuntimeError(f"{tag}: {len(hits)} candidates for {what}: {hits}")
    return hits[0] if hits else None


def find_sessions(root):
    """All sessions below root, ordered by date; a list of SessionFiles.

    Sessions are defined by the suite2p_output folders. A file kind that cannot be
    found is None in .files (see SessionFiles.missing); more than one candidate
    raises.
    """
    root = os.path.abspath(os.path.expanduser(root))
    s2p_root = os.path.join(root, "suite2p_output")
    if not os.path.isdir(s2p_root):
        raise FileNotFoundError(f"no suite2p_output folder in {root}")
    sessions = []
    for d in sorted(os.listdir(s2p_root)):
        m = TAG_RE.match(d)
        if d.startswith(".") or not m or not os.path.isdir(os.path.join(s2p_root, d)):
            continue
        tag = m.group(0)
        animal, date, time = m.group("animal"), m.group("date"), m.group("time")
        a = os.path.join(root, "aligned_arrays")
        activity = _one(os.path.join(a, "neuronal_activity", f"*_session_*_{tag}*.npy"), "activity", tag)
        k = re.search(r"_session_(\d+)_", os.path.basename(activity)) if activity else None
        files = {
            "suite2p": os.path.join(s2p_root, d),
            "activity": activity,
            "xy": _one(os.path.join(a, "XY_position", f"XY_raw_{tag}.npy"), "xy", tag),
            "location": _one(os.path.join(a, "maze_location", f"Location_raw_{tag}.npy"), "location", tag),
            "trial_times": _one(os.path.join(a, "trial_times", f"trialtimes_{tag}.npy"), "trial_times", tag),
            "task": _one(os.path.join(a, "tasks_persess", f"Task_data_{tag}.npy"), "task", tag),
            "mean_chan1": _one(os.path.join(root, "masks", f"{tag}*_mean_chan1.tif"), "mean_chan1", tag),
            "mean_chan2": _one(os.path.join(root, "masks", f"{tag}*_mean_chan2.tif"), "mean_chan2", tag),
            "seg_chan1": _one(os.path.join(root, "masks", f"{tag}*_mean_chan1_seg.npy"), "seg_chan1", tag),
            "seg_chan2": _one(os.path.join(root, "masks", f"{tag}*_mean_chan2_seg.npy"), "seg_chan2", tag),
            "h5": _one(os.path.join(root, "processed_funcimg", f"file_*_{date}_{time}.h5"), "h5", tag),
            "tsv": _one(os.path.join(root, "raw_behaviour", f"{animal}-{date}-*.tsv"), "tsv", tag),
            "video": _one(os.path.join(root, "raw_video", f"{animal}_{date}-??????.mp4"), "video", tag),
            "gpio": _one(os.path.join(root, "raw_video", f"{animal}_{date}-*_GPIO_data.csv"), "gpio", tag),
        }
        sessions.append(SessionFiles(tag, animal, date, time, int(k.group(1)) if k else None, files))
    if not sessions:
        raise FileNotFoundError(f"no sessions in {s2p_root}")
    return sorted(sessions, key=lambda s: (s.date, s.time))


def select_session(sessions, key):
    """The session whose tag, label or date contains key (must be unique)."""
    hits = [s for s in sessions if key in s.tag or key in s.label]
    if len(hits) != 1:
        raise KeyError(f"{key!r} matches {len(hits)} sessions: {[s.label for s in hits]}")
    return hits[0]
