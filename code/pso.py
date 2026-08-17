"""Particle Swarm Optimisation for Robot Path Planning (discrete adaptation)."""
import numpy as np
from grid import (make_grid, astar, is_valid_path, path_length, path_turns)
from ga import fitness, perturb, crossover_ncp, simplify, repair, init_population

# ponytail: velocity as crossover-based update (not edit-set / PSO-D).
#           Simpler, works well in practice. Upgrade to edit-set if fidelity matters.
# Ref: MDPI Sensors 2023 — PSO with Artificial Potential Fields for RPP

def run_pso(grid, start, goal, n_particles=80, n_iter=500,
            w=0.5, c1=1.5, c2=1.5, seed=0):
    """
    Discrete PSO for grid path planning.
    w  = inertia weight (prob of keeping current path in crossover)
    c1 = cognitive weight (prob of crossover toward personal best)
    c2 = social weight   (prob of crossover toward global best)
    Returns (best_path, fitness_history).
    """
    rng = np.random.default_rng(seed)
    n = grid.shape[0]

    particles = init_population(grid, start, goal, n_particles, rng)
    fits = [fitness(p, grid) for p in particles]
    pbest = [p.copy() for p in particles]
    pbest_fits = fits.copy()

    gbest_idx = int(np.argmin(fits))
    gbest = particles[gbest_idx].copy()
    gbest_fit = fits[gbest_idx]

    history = [gbest_fit]

    for _ in range(n_iter):
        for i in range(n_particles):
            p = particles[i].copy()

            # cognitive pull: crossover toward personal best
            r1 = rng.random()
            if r1 < c1:
                p = crossover_ncp(p, pbest[i], rng)

            # social pull: crossover toward global best
            r2 = rng.random()
            if r2 < c2:
                p = crossover_ncp(p, gbest, rng)

            # inertia: with prob w, also try original (take best of new vs old)
            # ponytail: inertia modelled as mutation rather than blend weight;
            #           proper inertia would weight the edit-set contribution.
            if rng.random() < w:
                p = perturb(p, grid, rng)

            p = simplify(p, n)
            p = repair(p, grid)

            f = fitness(p, grid)
            particles[i] = p

            if f < pbest_fits[i]:
                pbest[i] = p.copy()
                pbest_fits[i] = f

            if f < gbest_fit:
                gbest = p.copy()
                gbest_fit = f

        history.append(gbest_fit)

    return gbest, history

if __name__ == "__main__":
    g = make_grid(15, 0.2, seed=0)
    start, goal = 0, 15*15-1
    path, hist = run_pso(g, start, goal, seed=0)
    assert is_valid_path(path, g), "PSO returned invalid path"
    ref = astar(g, start, goal)
    n = g.shape[0]
    print(f"PSO 15x15: length={path_length(path,n):.2f} turns={path_turns(path,n)}")
    print(f"A*  15x15: length={path_length(ref,n):.2f}  turns={path_turns(ref,n)}")
