"""V_MINSPANE - minimum spanning tree using Euclidean distance."""

import numpy as np
from scipy.spatial import Delaunay


def v_minspane(x):
    """Calculate minimum spanning tree using Euclidean distance.

    Uses Delaunay triangulation to find candidate edges, then applies
    Kruskal's algorithm.

    Parameters
    ----------
    x : array_like, shape (n, d)
        d-dimensional data points.

    Returns
    -------
    p : ndarray, shape (n-1,)
        Parent node indices (1-indexed). Node n is the root.
    s : ndarray, shape (n-1,)
        Edge indices sorted by ascending Euclidean distance (1-indexed).
    """
    x = np.asarray(x, dtype=float)
    np_pts, nd = x.shape

    # Delaunay triangulation
    tri = Delaunay(x)
    simplices = tri.simplices  # (nt, nd+1)

    # Extract all edges from simplices
    from itertools import combinations
    edges = set()
    for simplex in simplices:
        for i, j in combinations(sorted(simplex), 2):
            edges.add((i, j))

    ee = np.array(sorted(edges))  # (ne, 2), 0-indexed
    ne = ee.shape[0]

    # Compute edge lengths squared
    sz = np.sum((x[ee[:, 0], :] - x[ee[:, 1], :]) ** 2, axis=1)
    mz = np.argsort(sz)
    ee = ee[mz, :]  # sort by ascending length

    # Kruskal's algorithm with union-find
    parent = np.full(np_pts, -1, dtype=int)  # -1 means root
    rank = np.zeros(np_pts, dtype=int)

    def find(i):
        while parent[i] >= 0:
            parent[i] = parent[parent[i]] if parent[parent[i]] >= 0 else parent[i]  # path compression
            i = parent[i]
        return i

    ei = []
    for i in range(ne):
        r1 = find(ee[i, 0])
        r2 = find(ee[i, 1])
        if r1 != r2:
            ei.append(i)
            # Union by rank
            if rank[r1] > rank[r2]:
                parent[r2] = r1
            elif rank[r1] < rank[r2]:
                parent[r1] = r2
            else:
                parent[r2] = r1
                rank[r1] += 1
            if len(ei) == np_pts - 1:
                break

    mst_edges = ee[ei, :]  # (n-1, 2), 0-indexed

    # Build tree with node (np_pts-1) as root (0-indexed), then convert to 1-indexed
    # Use BFS from root
    from collections import defaultdict, deque
    adj = defaultdict(list)
    edge_map = {}
    for idx, (u, v) in enumerate(mst_edges):
        adj[u].append(v)
        adj[v].append(u)
        edge_map[(min(u, v), max(u, v))] = idx

    root = np_pts - 1  # 0-indexed root
    p = np.zeros(np_pts - 1, dtype=int)  # parent array (1-indexed output)
    visited = np.zeros(np_pts, dtype=bool)
    visited[root] = True
    queue = deque([root])

    edge_lengths = np.sqrt(sz[np.array(ei)])

    # Build parent relationships and edge lengths for sorting
    node_edge_len = np.zeros(np_pts - 1)

    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                p[neighbor if neighbor < np_pts - 1 else neighbor] = node + 1  # 1-indexed parent
                if neighbor < np_pts - 1:
                    p[neighbor] = node + 1
                    key = (min(node, neighbor), max(node, neighbor))
                    edge_idx = edge_map[key]
                    node_edge_len[neighbor] = sz[ei[edge_idx]]
                queue.append(neighbor)

    # Sort edges by ascending length
    # s gives nodes sorted by edge length to parent
    lengths = np.sqrt(np.sum((x[:np_pts - 1, :] - x[p - 1, :]) ** 2, axis=1))
    s = np.argsort(lengths) + 1  # 1-indexed

    return p, s
