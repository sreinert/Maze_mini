# Maze_mini

Analysis of miniature two-photon microscope recordings from mice freely navigating a maze while performing several tasks.

## Structure

- `preprocessing/` – preprocessing steps (e.g. destriping, motion correction, mean images)
- `quality_control/` – sanity checks and QC routines (read-only on the data)
- `old_testing/` – earlier exploratory notebooks

Raw data is never modified: see `CLAUDE.md` for the data rules and repo conventions.

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
