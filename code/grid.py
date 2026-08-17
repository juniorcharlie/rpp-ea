"""Grid model, A* baseline, path validation, and path metrics."""
import numpy as np
import heapq
import math

# 8-directional Moore neighbourhood moves: (dr, dc)
MOVES = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
MOVE_COSTS = [math.sqrt(dr**2 + dc**2) for dr, dc in MOVES]

def make_grid(n, obstacle_rate, seed=0):
    """Return NxN binary grid (0=free, 1=obstacle). Start/goal always free."""
    rng = np.random.default_rng(seed)
    g = (rng.random((n, n)) < obstacle_rate).astype(np.int8)
    g[0, 0] = 0
    g[n-1, n-1] = 0
    return g

def cell_to_rc(cell, n):
    return divmod(cell, n)

def rc_to_cell(r, c, n):
    return r * n + c

def build_neighbour_cache(grid):
    """Precompute neighbour lists + costs for every cell."""
    n = grid.shape[0]
    cache = [None] * (n * n)
    for r in range(n):
        for c in range(n):
            cell = r * n + c
            if grid[r, c] == 1:
                cache[cell] = []
                continue
            nbrs = []
            for mi, (dr, dc) in enumerate(MOVES):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == 0:
                    nbrs.append((nr * n + nc, MOVE_COSTS[mi]))
            cache[cell] = nbrs
    return cache

# Cache keyed by grid content hash — safe across garbage collection cycles
# ponytail: tobytes() hash is O(n^2) per call but only on cache miss;
#           fine for grids up to 100x100. Use joblib memory for larger grids.
_nbr_cache = {}

def _get_cache(grid):
    key = (grid.shape, grid.tobytes())
    if key not in _nbr_cache:
        if len(_nbr_cache) > 100:
            _nbr_cache.clear()  # prevent unbounded growth
        _nbr_cache[key] = build_neighbour_cache(grid)
    return _nbr_cache[key]

# Heuristic cache keyed by (goal, n) — correct since h only depends on goal position
_h_cache = {}

def _build_h(goal, n):
    key = (goal, n)
    if key not in _h_cache:
        gr, gc = divmod(goal, n)
        rows = np.arange(n * n) // n
        cols = np.arange(n * n) % n
        _h_cache[key] = np.sqrt((rows - gr) ** 2 + (cols - gc) ** 2)
    return _h_cache[key]

def neighbours(cell, grid):
    """Return free neighbour cells (8-connected)."""
    return [c for c, _ in _get_cache(grid)[cell]]

def move_cost(a, b, n):
    r1, c1 = divmod(a, n)
    r2, c2 = divmod(b, n)
    return math.sqrt((r1-r2)**2 + (c1-c2)**2)

def astar(grid, start, goal):
    """Return shortest path as cell-sequence list, or None if unreachable."""
    cache = _get_cache(grid)
    h = _build_h(goal, grid.shape[0])

    dist = {start: 0.0}
    prev = {}
    visited = set()
    pq = [(h[start], start)]
    while pq:
        _, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == goal:
            path = []
            while u in prev:
                path.append(u)
                u = prev[u]
            path.append(start)
            return path[::-1]
        du = dist[u]
        for v, cost in cache[u]:
            d = du + cost
            if d < dist.get(v, math.inf):
                dist[v] = d
                prev[v] = u
                heapq.heappush(pq, (d + h[v], v))
    return None

def path_length(path, n):
    return sum(move_cost(path[i], path[i+1], n) for i in range(len(path)-1))

def path_turns(path, n):
    if len(path) < 3:
        return 0
    turns = 0
    for i in range(1, len(path)-1):
        r0,c0 = divmod(path[i-1], n)
        r1,c1 = divmod(path[i],   n)
        r2,c2 = divmod(path[i+1], n)
        if (r1-r0, c1-c0) != (r2-r1, c2-c1):
            turns += 1
    return turns

def is_valid_path(path, grid):
    if len(path) < 2:
        return False
    n = grid.shape[0]
    cache = _get_cache(grid)
    nbr_sets = {cell: {c for c, _ in cache[cell]} for cell in path}
    for cell in path:
        r, c = divmod(cell, n)
        if not (0 <= r < n and 0 <= c < n) or grid[r, c] == 1:
            return False
    for i in range(len(path)-1):
        if path[i+1] not in nbr_sets[path[i]]:
            return False
    return True

if __name__ == "__main__":
    g = make_grid(15, 0.2, seed=0)
    start, goal = 0, 15*15-1
    path = astar(g, start, goal)
    assert path and is_valid_path(path, g), "A* failed"
    print(f"A* 15x15: length={path_length(path,15):.2f} turns={path_turns(path,15)}")
