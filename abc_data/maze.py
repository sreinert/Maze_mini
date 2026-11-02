"""
Maze geometry and location ids of the ABC task.

The maze is a 3x3 grid of nodes (each with a poke port) joined by 12 edges.
Location ids in maze_location (Location_raw_*.npy):

    -1        unknown (no tracking)
    1..9      nodes, row by row from the bottom of the video image
              (1 2 3 / 4 5 6 / 7 8 9, x increasing to the right)
    10..21    edges, see EDGES

Port names in pyControl are <column letter><row number>: node n is port
"ABC"[(n-1) % 3] + str((n-1)//3 + 1), so 1=A1, 2=B1, 3=C1, 4=A2, ... 9=C3.
This was read off the pokes that triggered A_on/B_on/C_on against active_poke
in all six AB_08 sessions (abc_behaviour_qc checks it per session). The edge ids
were read off the tracked positions (edge centroid between its two nodes).
In the AB_08 maze node 9 (top right in the video) is a platform with no bridges:
edges 11 (8-9) and 14 (6-9) are absent (visible in the video, and neither ever
occurs in maze_location), so shortest paths exclude them (ABSENT_EDGES).
"""
from collections import deque

import numpy as np

NODES = tuple(range(1, 10))
EDGES = {10: (7, 8), 11: (8, 9), 12: (4, 7), 13: (5, 8), 14: (6, 9), 15: (4, 5),
         16: (5, 6), 17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (1, 2), 21: (2, 3)}
LOCATIONS = NODES + tuple(EDGES)
UNKNOWN = -1
ABSENT_EDGES = (11, 14)


def node_to_port(n):
    return "ABC"[(n - 1) % 3] + str((n - 1) // 3 + 1)


def port_to_node(port):
    return "ABC".index(port[0]) + 1 + 3 * (int(port[1]) - 1)


def is_node(loc):
    return np.isin(loc, NODES)


def location_name(loc):
    loc = int(loc)
    if loc in EDGES:
        a, b = EDGES[loc]
        return f"{a}-{b}"
    return str(loc) if loc in NODES else "?"


def node_distances(absent=ABSENT_EDGES):
    """(10, 10) shortest-path lengths in edges between nodes 1..9 (index = node id;
    inf = unreachable), leaving out the edges in absent."""
    adj = {n: [] for n in NODES}
    for e, (a, b) in EDGES.items():
        if e in absent:
            continue
        adj[a].append(b)
        adj[b].append(a)
    d = np.full((10, 10), np.inf)
    for s in NODES:
        d[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[s, v] == np.inf:
                    d[s, v] = d[s, u] + 1
                    q.append(v)
    return d


def node_path(location):
    """Sequence of distinct nodes visited (consecutive repeats and edges/unknown
    dropped)."""
    loc = np.asarray(location)
    nodes = loc[is_node(loc)].astype(int)
    if not len(nodes):
        return nodes
    return nodes[np.r_[True, nodes[1:] != nodes[:-1]]]


def location_centroids(xy, location):
    """Median xy per location id (dict); from tracking, for plotting and checks."""
    out = {}
    for l in LOCATIONS:
        m = (location == l) & np.all(np.isfinite(xy), axis=1)
        if m.any():
            out[l] = np.median(xy[m], axis=0)
    return out


def _clusters_1d(v, gap):
    """Indices of v grouped where sorted values are more than gap apart, the
    three groups with most members, ordered by value."""
    o = np.argsort(v)
    groups = np.split(o, np.flatnonzero(np.diff(v[o]) > gap) + 1)
    groups = sorted(groups, key=len, reverse=True)[:3]
    return sorted(groups, key=lambda g: np.median(v[g]))


def node_centres_from_background(bg, scale=1, n_candidates=20, tol=50):
    """(10, 2) video px of the port LED of each node (row = node id, row 0 unused)
    from a background image. Candidates: the largest bright blobs (top 0.5 % of
    pixels); kept if another candidate lines up with it in x and another in y
    (within tol video px), which drops reflections outside the grid; then
    clustered into 3 columns and 3 rows (node 1 bottom left, y down in the
    image), largest blob per cell. A node without a blob (LED dim in that
    session) gets its column's median x and row's median y."""
    import cv2
    b = cv2.GaussianBlur(np.asarray(bg, np.float32), (0, 0), 2)
    n, _, st, cen = cv2.connectedComponentsWithStats((b > np.percentile(b, 99.5)).astype(np.uint8))
    order = 1 + np.argsort(-st[1:, cv2.CC_STAT_AREA])[:n_candidates]
    pts, area = cen[order] * scale, st[order, cv2.CC_STAT_AREA]
    dx = np.abs(pts[:, None, 0] - pts[None, :, 0]) < tol
    dy = np.abs(pts[:, None, 1] - pts[None, :, 1]) < tol
    np.fill_diagonal(dx, False)
    np.fill_diagonal(dy, False)
    keep = dx.any(1) & dy.any(1)
    pts, area = pts[keep], area[keep]
    gap = 3 * tol
    cols, rows = _clusters_1d(pts[:, 0], gap), _clusters_1d(pts[:, 1], gap)[::-1]  # bottom row first
    if len(cols) < 3 or len(rows) < 3:
        raise RuntimeError(f"could not find a 3x3 grid of port LEDs ({len(pts)} aligned blobs)")
    col_of, row_of = np.full(len(pts), -1), np.full(len(pts), -1)
    for k, g in enumerate(cols):
        col_of[g] = k
    for k, g in enumerate(rows):
        row_of[g] = k
    out = np.full((10, 2), np.nan)
    for r in range(3):
        for c in range(3):
            m = np.flatnonzero((row_of == r) & (col_of == c))
            if len(m):
                out[1 + 3 * r + c] = pts[m[np.argmax(area[m])]]
    for r in range(3):
        for c in range(3):
            if np.isnan(out[1 + 3 * r + c, 0]):
                out[1 + 3 * r + c] = (np.nanmedian(out[1 + c::3][:3, 0]), np.nanmedian(out[1 + 3 * r:4 + 3 * r, 1]))
    if np.isnan(out[1:]).any() or np.sum([np.isfinite(out[1 + k, 0]) for k in range(9)]) < 9:
        raise RuntimeError("incomplete LED grid")
    return out


def node_spacing(centres):
    """Median distance between nodes joined by an edge (px)."""
    return float(np.median([np.linalg.norm(centres[a] - centres[b]) for a, b in EDGES.values()]))


def classify_positions(xy, centres, node_radius=0.3, edge_width=0.3, absent=ABSENT_EDGES):
    """Location id per position (n, 2): a node if within node_radius * spacing of
    its centre, else the nearest present edge if within edge_width * spacing of
    its segment, else UNKNOWN (also for NaN positions). Against Arya's labels
    (from head XY) this agrees in 93-94 % of frames from our body-centre track
    (all disagreements node vs edge at the platform borders)."""
    xy = np.asarray(xy, float)
    sp = node_spacing(centres)
    C = centres[1:]
    d = np.linalg.norm(xy[:, None] - C[None], axis=2)
    loc = np.full(len(xy), UNKNOWN)
    at = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1) < node_radius * sp
    loc[at] = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=1)[at] + 1
    ids = [e for e in EDGES if e not in absent]
    D = []
    for e in ids:
        a, b = centres[EDGES[e][0]], centres[EDGES[e][1]]
        t = np.clip(((xy - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
        D.append(np.linalg.norm(xy - (a + t[:, None] * (b - a)), axis=1))
    D = np.where(np.isfinite(D), D, np.inf)
    on_edge = ~at & (D.min(0) < edge_width * sp)
    loc[on_edge] = np.array(ids)[D.argmin(0)[on_edge]]
    loc[~np.all(np.isfinite(xy), axis=1)] = UNKNOWN
    return loc
