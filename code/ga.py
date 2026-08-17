"""Genetic Algorithm for Robot Path Planning."""
import numpy as np
import math
from grid import (make_grid, astar, neighbours, path_length, path_turns,
                  is_valid_path, move_cost, cell_to_rc, rc_to_cell, _get_cache)

# Fitness weights
W1, W2, W3 = 0.4, 0.2, 100.0

def fitness(path, grid):
    n = grid.shape[0]
    L = path_length(path, n)
    T = path_turns(path, n)
    C = sum(1 for cell in path if grid[cell // n, cell % n] == 1)
    return W1 * L + W2 * T + W3 * C

def simplify(path, n):
    """Remove collinear waypoints (same direction as prev→next). Reduces turns for free."""
    if len(path) < 3:
        return path
    out = [path[0]]
    for i in range(1, len(path)-1):
        r0,c0 = cell_to_rc(out[-1], n)
        r1,c1 = cell_to_rc(path[i], n)
        r2,c2 = cell_to_rc(path[i+1], n)
        d0 = (r1-r0, c1-c0)
        d1 = (r2-r1, c2-c1)
        if d0 != d1:
            out.append(path[i])
    out.append(path[-1])
    return out

def repair(path, grid):
    """Reroute invalid segments via local A*. Always ends at path[-1] (goal)."""
    if is_valid_path(path, grid):
        return path
    n = grid.shape[0]
    goal = path[-1]
    fixed = [path[0]]
    i = 0
    while i < len(path) - 1:
        a, b = path[i], path[i+1]
        if b in neighbours(a, grid):
            fixed.append(b)
            i += 1
        else:
            # Reroute: try each remaining waypoint including goal
            for j in range(i+2, len(path)):
                seg = astar(grid, a, path[j])
                if seg:
                    fixed.extend(seg[1:])
                    i = j
                    break
            else:
                # Can't reach any remaining waypoint; go straight to goal
                seg = astar(grid, a, goal)
                if seg:
                    fixed.extend(seg[1:])
                break
    # Ensure goal is reached
    if not fixed or fixed[-1] != goal:
        seg = astar(grid, fixed[-1] if fixed else path[0], goal)
        if seg:
            fixed.extend(seg[1:])
    return fixed if is_valid_path(fixed, grid) else path  # ponytail: keep original if repair fails; collision penalty handles it

def perturb(path, grid, rng, delta=2):
    """Range mutation: random-probe insert or delete — O(1) per call.
    Probes one random edge for insert, one random cell for delete.
    Falls back to the other op if the probe yields nothing.
    # ponytail: one probe only; full O(n) scan would find more candidates
    #           but dominates runtime on paths >50 cells.
    """
    if len(path) < 3:
        return path

    cache = _get_cache(grid)

    if rng.random() < 0.5:
        # Try insert: pick a random edge, find a shared neighbour
        idx = int(rng.integers(0, len(path) - 1))
        a, b = path[idx], path[idx+1]
        nb_b = {c for c, _ in cache[b]}
        shared = [c for c, _ in cache[a] if c != b and c in nb_b]
        if shared:
            ins = shared[int(rng.integers(len(shared)))]
            return path[:idx+1] + [ins] + path[idx+1:]
        # Fallback: try delete
        idx = int(rng.integers(1, len(path) - 1))
        if path[idx+1] in {c for c, _ in cache[path[idx-1]]}:
            return path[:idx] + path[idx+1:]
    else:
        # Try delete: pick a random interior cell
        idx = int(rng.integers(1, len(path) - 1))
        if path[idx+1] in {c for c, _ in cache[path[idx-1]]}:
            return path[:idx] + path[idx+1:]
        # Fallback: try insert
        idx = int(rng.integers(0, len(path) - 1))
        a, b = path[idx], path[idx+1]
        nb_b = {c for c, _ in cache[b]}
        shared = [c for c, _ in cache[a] if c != b and c in nb_b]
        if shared:
            ins = shared[int(rng.integers(len(shared)))]
            return path[:idx+1] + [ins] + path[idx+1:]

    return path  # no valid mutation found this probe

def crossover_ncp(p1, p2, rng):
    """Non-common-point crossover: splice at a shared interior waypoint.
    Takes prefix of p1 up to shared point, then suffix of p2 FROM last occurrence
    of that point to end — guarantees child ends at goal (p2[-1]).
    """
    shared = set(p1[1:-1]) & set(p2[1:-1])
    if not shared:
        return p1.copy()  # ponytail: no shared point → return p1 unchanged
    pt = int(rng.choice(list(shared)))
    i1 = p1.index(pt)                  # first occurrence in p1
    i2 = len(p2) - 1 - p2[::-1].index(pt)  # last occurrence in p2 → suffix reaches goal
    return p1[:i1] + p2[i2:]

def tournament(pop, fits, k, rng):
    """Tournament selection: pick best of k random individuals."""
    idxs = rng.choice(len(pop), k, replace=False)
    return pop[int(idxs[np.argmin([fits[i] for i in idxs])])]

def _random_directional_path(grid, start, goal, rng):
    """Directionally-biased random walk (Zhu & Pan [1]-style): at each step,
    samples among free unvisited neighbours weighted by inverse distance to
    the goal. Never calls A* itself. If it dead-ends before reaching the
    goal, appends the goal cell so the caller's repair() reroutes the
    broken tail exactly like it reroutes any other invalid child.
    """
    n = grid.shape[0]
    gr, gc = cell_to_rc(goal, n)
    path = [start]
    visited = {start}
    current = start
    for _ in range(n * n * 2):
        if current == goal:
            return path
        nbrs = [c for c in neighbours(current, grid) if c not in visited]
        if not nbrs:
            break
        w = np.empty(len(nbrs))
        for i, c in enumerate(nbrs):
            r, cc = cell_to_rc(c, n)
            w[i] = 1.0 / (math.hypot(r - gr, cc - gc) + 1e-6)
        cumw = np.cumsum(w)
        nxt = nbrs[int(np.searchsorted(cumw, rng.random() * cumw[-1]))]
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    if path[-1] != goal:
        path.append(goal)
    return path

def init_population(grid, start, goal, size, rng, use_astar_init=True):
    """Seed population from a single A* baseline + perturbations
    (use_astar_init=True, original behaviour) or from independently
    constructed directionally-biased random walks (use_astar_init=False),
    each repaired to validity. repair() itself still calls A* locally for
    broken segments in both branches — that's shared machinery held
    constant across the ablation, not part of what this flag controls.
    """
    if use_astar_init:
        base = astar(grid, start, goal)
        if base is None:
            raise ValueError("No path from start to goal (A* failed).")
        pop = [base]
        attempts = 0
        while len(pop) < size and attempts < size * 20:
            cand = base.copy()
            for _ in range(rng.integers(1, 5)):
                cand = perturb(cand, grid, rng)
            cand = repair(cand, grid)
            pop.append(cand)
            attempts += 1
        while len(pop) < size:
            pop.append(base.copy())
        return pop

    pop = []
    attempts = 0
    while len(pop) < size and attempts < size * 20:
        cand = repair(_random_directional_path(grid, start, goal, rng), grid)
        pop.append(cand)
        attempts += 1
    if not pop:
        raise ValueError("Random-directional init produced no individuals.")
    while len(pop) < size:
        pop.append(pop[int(rng.integers(len(pop)))].copy())
    return pop

def run_ga(grid, start, goal, pop_size=80, n_gen=500, pc=0.8, pm=0.05, k=3,
           seed=0, use_astar_init=True, use_elitism=True, adaptive_pm=False,
           reinject_frac=0.0, stagnation_window=10, pm_boost=4.0, pm_cap=0.3):
    """
    Returns (best_path, fitness_history).
    Ablation flags (all default to the original Phase-1 GA behaviour):
      use_astar_init  - seed from A* baseline vs. random-directional walks
      use_elitism     - carry top 5% forward unchanged each generation
      adaptive_pm     - after `stagnation_window` gens with no improvement,
                        boost mutation prob to min(pm*pm_boost, pm_cap)
      reinject_frac   - while stagnated, replace this fraction of the worst
                        offspring with fresh individuals each generation
    """
    rng = np.random.default_rng(seed)
    n = grid.shape[0]
    elite_n = max(1, int(pop_size * 0.05)) if use_elitism else 0

    pop = init_population(grid, start, goal, pop_size, rng, use_astar_init=use_astar_init)
    fits = [fitness(p, grid) for p in pop]
    history = [min(fits)]
    best_so_far = min(fits)
    stagn = 0

    for _ in range(n_gen):
        order = np.argsort(fits)
        new_pop = [pop[i] for i in order[:elite_n]] if elite_n else []

        cur_best = fits[int(order[0])]
        if cur_best < best_so_far - 1e-9:
            best_so_far = cur_best
            stagn = 0
        else:
            stagn += 1
        stagnated = stagn >= stagnation_window
        pm_eff = min(pm * pm_boost, pm_cap) if (adaptive_pm and stagnated) else pm

        while len(new_pop) < pop_size:
            p1 = tournament(pop, fits, k, rng)
            if rng.random() < pc:
                p2 = tournament(pop, fits, k, rng)
                child = crossover_ncp(p1, p2, rng)
            else:
                child = p1.copy()
            if rng.random() < pm_eff:
                child = perturb(child, grid, rng)
            child = simplify(child, n)
            child = repair(child, grid)
            new_pop.append(child)

        new_fits = [fitness(p, grid) for p in new_pop]

        if reinject_frac > 0 and stagnated:
            n_reinject = max(1, int(pop_size * reinject_frac))
            worst = np.argsort(new_fits)[-n_reinject:]
            fresh = init_population(grid, start, goal, n_reinject, rng, use_astar_init=use_astar_init)
            for slot, ind in zip(worst, fresh):
                new_pop[int(slot)] = ind
                new_fits[int(slot)] = fitness(ind, grid)

        pop, fits = new_pop, new_fits
        history.append(min(fits))

    best = pop[int(np.argmin(fits))]
    return best, history

if __name__ == "__main__":
    g = make_grid(15, 0.2, seed=0)
    start, goal = 0, 15*15-1
    path, hist = run_ga(g, start, goal, pop_size=50, n_gen=200, seed=0)
    assert is_valid_path(path, g), "GA returned invalid path"
    ref = astar(g, start, goal)
    n = g.shape[0]
    print(f"GA  15x15: length={path_length(path,n):.2f} turns={path_turns(path,n)}")
    print(f"A*  15x15: length={path_length(ref,n):.2f}  turns={path_turns(ref,n)}")
