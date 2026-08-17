"""Ant Colony Optimisation for Robot Path Planning (grid adaptation)."""
import numpy as np
import math
from grid import (make_grid, astar, neighbours, path_length, path_turns,
                  is_valid_path, cell_to_rc, _get_cache)
from ga import fitness, simplify, perturb, repair

# ponytail: pheromone per cell (not per edge) → N*N vs N*N*8 memory.
#           Upgrade to edge pheromone if quality is poor on sparse maps.
# ponytail: no ACS daemon actions / local search. Add 2-opt if quality falls short.
# Ref: MDPI Drones 2023 — Review of Autonomous Path Planning for Mobile Robots

def _build_eta(grid, goal):
    """Precompute heuristic (1/Chebyshev) for every cell — avoids per-step recompute."""
    n = grid.shape[0]
    gr, gc = cell_to_rc(goal, n)
    eta = np.zeros(n * n)
    for cell in range(n * n):
        r, c = cell_to_rc(cell, n)
        cheb = max(abs(r - gr), abs(c - gc))
        eta[cell] = 1.0 / (cheb + 1e-6)
    return eta

def _ant_walk(cache, eta, tau, alpha, beta, start, goal, n, rng):
    """Single ant walk. Uses precomputed eta and neighbour cache for speed."""
    path = [start]
    visited = {start}
    current = start
    max_steps = n * n * 2  # ponytail: safety cap; real ACO uses cycle detection

    for _ in range(max_steps):
        if current == goal:
            return path
        # Filter: unvisited free neighbours
        nbrs = [c for c, _ in cache[current] if c not in visited]
        if not nbrs:
            return None
        # Weights: tau^alpha * eta^beta  (vectorised for speed)
        w = np.array([(tau[c] ** alpha) * (eta[c] ** beta) for c in nbrs])
        w_sum = w.sum()
        if w_sum == 0:
            w = np.ones(len(nbrs))
            w_sum = float(len(nbrs))
        # Roulette wheel via searchsorted (faster than rng.choice(p=...))
        cumw = np.cumsum(w)
        r = rng.random() * w_sum
        chosen = nbrs[int(np.searchsorted(cumw, r))]
        path.append(chosen)
        visited.add(chosen)
        current = chosen

    return None  # exceeded max steps

def run_aco(grid, start, goal, n_ants=50, n_iter=500,
            alpha=1.0, beta=2.0, rho=0.1, Q=1.0, seed=0,
            restart_dead_ends=False):
    """
    ACO for grid RPP.
    alpha = pheromone weight
    beta  = heuristic weight
    rho   = evaporation rate
    Q     = pheromone deposit constant
    restart_dead_ends - ablation flag (default False reproduces the
    original results exactly). When True, an ant that dead-ends
    before reaching the goal is replaced by a perturbed-from-current-
    best walk instead of being discarded — the GA's stagnation fix
    (perturb + repair) transplanted onto ACO's per-iteration
    construction step
    Returns (best_path, fitness_history).
    """
    rng = np.random.default_rng(seed)
    n = grid.shape[0]
    cache = _get_cache(grid)
    eta = _build_eta(grid, goal)

    tau = np.ones(n * n)  # uniform initial pheromone

    # Warm-start: deposit pheromone along A* path to avoid cold-start slowness
    ref = astar(grid, start, goal)
    if ref is None:
        raise ValueError("No path from start to goal (A* failed).")
    L_ref = path_length(ref, n)
    for cell in ref:
        tau[cell] += Q / L_ref

    best_path = ref
    best_fit = fitness(ref, grid)
    history = [best_fit]

    for _ in range(n_iter):
        paths = []
        for _ in range(n_ants):
            p = _ant_walk(cache, eta, tau, alpha, beta, start, goal, n, rng)
            if p and p[-1] == goal:
                p = simplify(p, n)
                paths.append(p)
            elif restart_dead_ends:
                p = repair(perturb(best_path, grid, rng), grid)
                if is_valid_path(p, grid):
                    paths.append(simplify(p, n))

        # Evaporate
        tau *= (1.0 - rho)
        np.clip(tau, 1e-6, None, out=tau)  # prevent zero pheromone

        # Deposit
        for p in paths:
            L = path_length(p, n)
            if L > 0:
                deposit = Q / L
                for cell in p:
                    tau[cell] += deposit

            f = fitness(p, grid)
            if f < best_fit:
                best_fit = f
                best_path = p

        history.append(best_fit)

    return best_path, history

if __name__ == "__main__":
    g = make_grid(15, 0.2, seed=0)
    start, goal = 0, 15 * 15 - 1
    path, hist = run_aco(g, start, goal, seed=0)
    assert is_valid_path(path, g), "ACO returned invalid path"
    ref = astar(g, start, goal)
    n = g.shape[0]
    print(f"ACO 15x15: length={path_length(path,n):.2f} turns={path_turns(path,n)}")
    print(f"A*  15x15: length={path_length(ref,n):.2f}  turns={path_turns(ref,n)}")
