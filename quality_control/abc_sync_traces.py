"""
Raw synchronisation signals of the ABC-task sessions, to look at rather than
summarise: the imaging DAQ traces (h5 FrameClock, PyControlPulse), the camera
GPIO lines and the pyControl rsync events.

    uv run python quality_control/abc_sync_traces.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding

One figure per session (sync_traces/<session>.png):

    row 1  whole recording on the DAQ clock: FrameClock as frames per second
           (1 s bins; a gap or a rate change would show), PyControlPulse raw
           trace, and the three rsync trains (DAQ, pyControl mapped through
           the clock fit, camera mapped through its fit) as rasters
    row 2  raw DAQ traces: first 5 s (DAQ start, first frame edges), +-1 s
           around the first rsync pulse, last 3 s (last frame edges)
    row 3  inter-pulse intervals of the rsync train in each recording, in
           pulse order: if the pulses are matched one to one the three
           sequences lie on top of each other (pyControl draws them at random,
           so a shift by one pulse would be obvious); camera GPIO1/2/3 raw
           around the first pulse; FrameClock pulse width / interval detail

Inputs are opened read-only.
"""
import h5py
import numpy as np

from abc_common import labels, new_figure, parser, safe, save, setup, title
from suite2p_io import GRIDLINE, INK_MUTED, SERIES_1, SERIES_2, SERIES_3, style_axis


def daq_raw(path):
    with h5py.File(path, "r") as f:
        sr = float(f.attrs["samplerate"])
        return sr, {k: f[k][:] for k in f.keys()}


def _trace(ax, t, x, lo, hi, color, label=None):
    m = (t >= lo) & (t <= hi)
    ax.plot(t[m], x[m], color=color, linewidth=0.8, label=label)
    ax.set_xlim(lo, hi)


def plot_session(s, out_path):
    sr, raw = daq_raw(s.files.files["h5"])
    fc, pc = raw["FrameClock"], raw["PyControlPulse"]
    t = np.arange(len(fc)) / sr
    sy, cam, log = s.imaging_sync, s.camera_sync, s.pycontrol
    daq, camfit = s.pyc_to_daq, s.pyc_to_camera
    rs_pyc = log.rsync_times()
    rs_pyc_daq = daq(rs_pyc)                                     # pyControl pulses on DAQ clock
    rs_cam_daq = daq((cam.pulse_times - camfit.intercept) / camfit.slope)   # camera pulses on DAQ clock
    ft = sy.frame_times[:sy.n_edges]

    fig, axes = new_figure(3, 3, figsize=(22, 13))
    for ax in axes.ravel():
        style_axis(ax)

    # row 1: whole recording
    ax = axes[0, 0]
    bins = np.arange(0, t[-1] + 1, 1.0)
    rate, _ = np.histogram(ft, bins)
    ax.step(bins[:-1], rate, where="post", color=SERIES_1, linewidth=1)
    ax.axvline(daq(0.0), color=SERIES_3, linewidth=1.5, label="pyControl t = 0")
    ax.axvline(daq(log.duration), color=SERIES_3, linewidth=1.5, linestyle="--", label="pyControl end")
    w0, w1 = ft[s.aligned_offset], ft[min(s.aligned_offset + s.n_frames - 1, len(ft) - 1)]
    ax.axvspan(w0, w1, color=SERIES_2, alpha=0.12, label="aligned window")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    ax.set_ylim(0, rate.max() * 1.25)
    title(ax, f"FrameClock rate ({sy.n_edges} edges, suite2p {s.n_s2p_frames} frames)")
    labels(ax, "DAQ time (s)", "frames per s")

    ax = axes[0, 1]
    step = max(1, int(sr / 200))
    ax.plot(t[::step], pc[::step], color=SERIES_2, linewidth=0.5)
    title(ax, f"PyControlPulse, whole recording ({len(sy.pulse_times)} pulses)")
    labels(ax, "DAQ time (s)", "V")

    ax = axes[0, 2]
    for y, (ev, name, c) in enumerate(((sy.pulse_times, "DAQ PyControlPulse", SERIES_1),
                                       (rs_pyc_daq, "pyControl rsync (fit)", SERIES_2),
                                       (rs_cam_daq, "camera GPIO1 (fit)", SERIES_3))):
        ax.plot(ev, np.full(len(ev), y), "|", color=c, markersize=14, markeredgewidth=1.5, label=name)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["DAQ", "pyControl", "camera"], fontsize=8)
    ax.set_ylim(-0.7, 2.7)
    ax.axvline(daq(0.0), color=INK_MUTED, linewidth=1)
    title(ax, "rsync pulse trains on the DAQ clock")
    labels(ax, "DAQ time (s)")

    # row 2: raw DAQ zooms
    zooms = ((0, 5, "DAQ start"),
             (sy.pulse_times[0] - 1, sy.pulse_times[0] + 1, "first rsync pulse"),
             (t[-1] - 3, t[-1], "DAQ end"))
    for ax, (lo, hi, name) in zip(axes[1], zooms):
        _trace(ax, t, fc, lo, hi, SERIES_1, "FrameClock")
        _trace(ax, t, pc, lo, hi, SERIES_2, "PyControlPulse")
        e = ft[(ft >= lo) & (ft <= hi)]
        ax.plot(e, np.full(len(e), fc.max() * 1.08), "v", color=SERIES_1, markersize=3, label="detected frame edges")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        title(ax, f"Raw DAQ: {name} ({lo:.2f}-{hi:.2f} s)")
        labels(ax, "DAQ time (s)", "V")

    # row 3
    ax = axes[2, 0]
    for ipi, name, c, mk in ((np.diff(sy.pulse_times), "DAQ", SERIES_1, "o"),
                             (np.diff(rs_pyc), "pyControl", SERIES_2, "s"),
                             (np.diff(cam.pulse_times), "camera", SERIES_3, "^")):
        ax.plot(np.arange(1, len(ipi) + 1), ipi, mk, color=c, markersize=6 if mk == "o" else 9,
                markerfacecolor=c if mk == "o" else "none", markeredgewidth=1.3, label=name)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    title(ax, "rsync inter-pulse intervals, in pulse order")
    labels(ax, "pulse number", "interval (s)")

    ax = axes[2, 1]
    ct = cam.frame_times
    c0 = cam.pulse_times[0]
    m = (ct > c0 - 1) & (ct < c0 + 1)
    for i, c in enumerate((SERIES_1, SERIES_2, SERIES_3)):
        ax.step(ct[m] - c0, cam.gpio[m, i] + 1.5 * i, where="post", color=c, linewidth=1.2, label=f"GPIO{i + 1}")
    ax.legend(frameon=False, fontsize=8)
    title(ax, "Camera GPIO around its first rsync pulse (one sample per video frame)")
    labels(ax, "time from first camera pulse (s)", "level (offset per line)")

    ax = axes[2, 2]
    hi_ = fc > (fc.max() + fc.min()) / 2
    up = np.flatnonzero(~hi_[:-1] & hi_[1:]) + 1
    down = np.flatnonzero(hi_[:-1] & ~hi_[1:]) + 1
    down = down[down > up[0]][:len(up)]
    width = (down - up[:len(down)]) / sr * 1e3
    ax.plot(ft[1:len(width)], np.diff(ft)[:len(width) - 1] * 1e3, ".", color=SERIES_1, markersize=2,
            label="edge-to-edge interval")
    ax.plot(ft[:len(width)], width, ".", color=SERIES_2, markersize=2, label="high duration")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, markerscale=4)
    title(ax, "FrameClock pulse timing over the recording")
    labels(ax, "DAQ time (s)", "ms")

    save(fig, out_path, f"{s.label}  |  raw sync signals  |  DAQ {t[-1]:.1f} s at {sr:g} Hz, "
                        f"pyControl {log.duration:.1f} s, video {cam.frame_times[-1]:.1f} s; "
                        f"aligned frame 0 = suite2p frame {s.aligned_offset}")


def main():
    a = parser(__doc__, "sync_traces").parse_args()
    root, out, sessions = setup(a, "sync_traces")
    for s in sessions:
        plot_session(s, out / f"{safe(s.label)}.png")
        print(f"{s.label}: done", flush=True)
    print(f"Saved figures to {out}")


if __name__ == "__main__":
    main()
