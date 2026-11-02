# Maze_mini

Analysis of miniature two-photon microscope recordings from mice freely navigating a maze while performing several tasks.

## Structure

- `preprocessing/` – preprocessing steps (e.g. destriping, motion correction, mean images)
- `quality_control/` – sanity checks and QC routines (read-only on the data)
- `abc_data/` – loaders for the ABC sequence-task data (Arya's preprocessed layout)
- `old_testing/` – earlier exploratory notebooks

Raw data is never modified: scripts only read the data and write their outputs elsewhere.

## Setup

```
uv sync
```

## Quality control

Both scripts take a path anywhere from a folder of sessions down to a single suite2p
`plane0` folder and process every suite2p plane below it. Output goes to
`~/Desktop/maze_mini_qc/<name of path>/` by default (`--out` to change).

### Registration

```
uv run python quality_control/registration_qc.py <preprocessed path>
```

One figure per session, one row per plane: registered mean images of both channels
(with the valid region after registration outlined), rigid x/y shifts, the
phase-correlation peak (corrXY) with bad frames marked, and non-rigid block shifts when
present. Works on sessions that were only registered (no detection/extraction yet).

### Signal quality

```
uv run python quality_control/suite2p_signal_qc.py <preprocessed path> [--neucoeff 0.7]
```

Per plane, the top-SNR and random cells' traces under the population mean; across
planes, per-cell baseline F, SNR, skew, event fraction, noise, cell-neuropil and
cell-population correlation, plus mean-image brightness and ROI counts. Metrics match
Bonsai_ABCD's `signal_quality_metrics.py`, so values are comparable with the ABCD sessions.

### Registered movie

```
uv run python quality_control/registered_movie.py <plane folder> [--start S] [--duration S] [--chan 1|2] [--speed X]
```

Sped-up mp4 of a time window of the registered stack, plus its mean projection, for
checking motion, bleaching, z-drift and dead or saturated periods. `--chan 1` is
suite2p's `data.bin` (functional channel), `--chan 2` is `data_chan2.bin`.
Frames are first copied to a local cache (`~/maze_mini_qc_cache`), so re-rendering
with other settings does not touch the share again.
`registered_movie.py` also takes an ABC extraction folder (`suite2p_output/<tag>_...`)
and renders its `green_channel.bin`.

## ABC sequence-task data

Loaders (`abc_data/`) and QC scripts (`quality_control/abc_*.py`) for the preprocessed
ABC-task sessions (freely moving mice, frontal cortex), e.g.
`/Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding`.

```python
from abc_data import find_sessions, load_session
s = load_session(find_sessions(root)[0])
s.activity, s.present, s.xy, s.location, s.trial_times, s.task, s.time, s.goal_progress(10)
```

Every script takes the folder as its argument (`--session 2026-08-13` to restrict it),
and writes to `~/Desktop/maze_mini_qc/<name of folder>/<kind>`:

| script | checks |
|---|---|
| `abc_inventory.py` | every count that must agree across files (frames, pulses, trials, ROIs) |
| `abc_sync_traces.py` | raw DAQ / camera / pyControl sync signals per session |
| `abc_sync_qc.py` | clock fits, frame rate, trial times vs pyControl, neural offset, location at pokes |
| `abc_roi_qc.py` | ROIs on mean images, ROICaT ↔ suite2p match, cross-session tracking consistency |
| `abc_signal_qc.py` | signal-quality metrics (as above), speed coupling, `--movie` frame correlation |
| `abc_behaviour_qc.py` | tracking, location labels, task performance, video frames at the pokes |
| `abc_neural_task_qc.py` | PETHs and spatial maps with cross-validated reliability |
| `abc_decoding_qc.py` | location / goal-progress decoding within and across sessions, lag scan |
| `abc_raw_qc.py` | our suite2p on raw sessions: sync counts, pyControl behaviour, PETHs |
| `abc_raw_lag_qc.py` | our suite2p on raw sessions: decoding vs lag between activity and behaviour |
| `abc_raw_tuning_qc.py`, `abc_processed_tuning_qc.py` | single-cell goal-progress tuning (raw / Arya's sessions) |
| `abc_spatial_tuning_qc.py` | spatial maps on the maze, within and across sessions |
| `abc_frame_order_qc.py` | frame order of Arya's processed movies vs our suite2p on the raw tif |

`preprocessing/track_position.py` tracks the mouse in the videos (location per imaging frame,
needed for location decoding and spatial maps on the raw sessions).

```
uv run python quality_control/abc_inventory.py /Volumes/behrens/Sandra/mini2p/Arya/PROCESSED/good_decoding
```
