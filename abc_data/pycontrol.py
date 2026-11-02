"""
Read pyControl (framework 2.x) .tsv logs.

Columns: time (s from task start), type (info/variable/print/state/event),
subtype (e.g. input, timer, sync) and content. In the ABC tasks:

    states   prime_A/prime_B/prime_C   waiting for a poke at the A/B/C port
             A_on/B_on/C_on            correct poke at that port (reward)
    events   <port>_in/_out            poke in/out, port names A1..C3 (see maze.py)
             rsync                     sync pulse, also recorded by the imaging
                                       DAQ (h5 PyControlPulse) and the camera (GPIO1)
    variable run_start/run_end         JSON with active_poke = [A, B, C] node ids
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PyControlLog:
    path: str
    df: pd.DataFrame
    info: dict
    run_start: dict

    @property
    def active_poke(self):
        return list(self.run_start.get("active_poke", []))

    def state_times(self, name):
        d = self.df
        return d.loc[(d.type == "state") & (d.content == name), "time"].to_numpy()

    def rsync_times(self):
        d = self.df
        return d.loc[(d.type == "event") & (d.content == "rsync"), "time"].to_numpy()

    def pokes(self):
        """Poke-in events: DataFrame with time and port (e.g. 'B3')."""
        d = self.df
        m = (d.type == "event") & (d.subtype == "input") & d.content.str.endswith("_in", na=False)
        out = d.loc[m, ["time"]].copy()
        out["port"] = d.loc[m, "content"].str[:-3]
        return out.reset_index(drop=True)

    @property
    def duration(self):
        t = self.df.loc[(self.df.type == "variable") & (self.df.subtype == "run_end"), "time"]
        return float(t.iloc[0]) if len(t) else float(self.df.time.max())


def load_pycontrol(path):
    df = pd.read_csv(path, sep="\t", dtype={"subtype": str, "content": str})
    df["subtype"] = df["subtype"].fillna("")
    info = dict(zip(df.loc[df.type == "info", "subtype"], df.loc[df.type == "info", "content"]))
    rs = df.loc[(df.type == "variable") & (df.subtype == "run_start"), "content"]
    run_start = json.loads(rs.iloc[0]) if len(rs) else {}
    return PyControlLog(path, df, info, run_start)


def trial_events(log):
    """Times (pyControl s) of A_on, B_on, C_on per completed trial and the next A_on
    (NaN after the last one), as an (n_trials, 4) array; trials = C_on events."""
    a, b, c = (log.state_times(n) for n in ("A_on", "B_on", "C_on"))
    rows = []
    for tc in c:
        ta = a[a < tc].max() if (a < tc).any() else np.nan
        tb = b[(b < tc) & (b > ta)].max() if ((b < tc) & (b > ta)).any() else np.nan
        nxt = a[a > tc]
        rows.append((ta, tb, tc, nxt.min() if len(nxt) else np.nan))
    return np.array(rows, dtype=float).reshape(-1, 4)
