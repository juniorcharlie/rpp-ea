"""Ablation experiments: which shared-framework component drives the
turn-count advantage over A* (Section 4 of the original paper), and does an
anti-stagnation fix built for GA transfer to ACO?

Reuses the benchmark matrix and per-grid-size hyperparameters from
experiments.py unchanged; only the GA/ACO configuration flags vary.
"""
import csv
import time
import os

from grid import make_grid, astar, path_length, path_turns
from ga import run_ga
from aco import run_aco
from experiments import INSTANCES, N_SEEDS, GA_PARAMS, ACO_PARAMS, _instance_label

GA_CONFIGS = {
    "full":              dict(use_astar_init=True,  use_elitism=True,  adaptive_pm=False, reinject_frac=0.0),
    "no_astar_init":     dict(use_astar_init=False, use_elitism=True,  adaptive_pm=False, reinject_frac=0.0),
    "no_elitism":        dict(use_astar_init=True,  use_elitism=False, adaptive_pm=False, reinject_frac=0.0),
    "adaptive_mut":      dict(use_astar_init=True,  use_elitism=True,  adaptive_pm=True,  reinject_frac=0.10),
    "adaptive_pm_only":  dict(use_astar_init=True,  use_elitism=True,  adaptive_pm=True,  reinject_frac=0.0),
    "reinject_only":     dict(use_astar_init=True,  use_elitism=True,  adaptive_pm=False, reinject_frac=0.10),
}

ACO_CONFIGS = {
    "baseline": dict(restart_dead_ends=False),
    "restart":  dict(restart_dead_ends=True),
}


def run_ga_ablation(output_csv="../datasets/raw/ga_ablation_results.csv",
                     instances=None, n_seeds=None, configs=None):
    instances = instances or INSTANCES
    n_seeds = N_SEEDS if n_seeds is None else n_seeds   
    configs = configs or GA_CONFIGS
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    rows = []
    total = len(instances) * n_seeds * len(configs)
    done = 0

    for grid_label, n, obs in instances:
        inst = _instance_label(grid_label, obs)
        gp = GA_PARAMS[n]
        for seed in range(n_seeds):
            g = make_grid(n, obs, seed=seed)
            start, goal = 0, n * n - 1
            ref = astar(g, start, goal)

            for cfg_name, cfg in configs.items():
                if ref is None:
                    rows.append({
                        "instance": inst, "seed": seed, "config": cfg_name,
                        "astar_length": "nan", "astar_turns": "nan",
                        "ga_length": "nan", "ga_turns": "nan", "runtime_ms": "nan",
                    })
                else:
                    t0 = time.perf_counter()
                    path, _ = run_ga(g, start, goal, seed=seed, pc=0.8, pm=0.05, k=3, **gp, **cfg)
                    ms = (time.perf_counter() - t0) * 1000
                    rows.append({
                        "instance": inst, "seed": seed, "config": cfg_name,
                        "astar_length": round(path_length(ref, n), 4),
                        "astar_turns": path_turns(ref, n),
                        "ga_length": round(path_length(path, n), 4),
                        "ga_turns": path_turns(path, n),
                        "runtime_ms": round(ms, 1),
                    })
                done += 1
                if done % 60 == 0:
                    print(f"  {done}/{total}  {inst} seed={seed} config={cfg_name}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {output_csv}")


def run_aco_ablation(output_csv="../datasets/raw/aco_ablation_results.csv",
                      instances=None, n_seeds=None):
    instances = instances or INSTANCES
    n_seeds = N_SEEDS if n_seeds is None else n_seeds
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    rows = []
    total = len(instances) * n_seeds * len(ACO_CONFIGS)
    done = 0

    for grid_label, n, obs in instances:
        inst = _instance_label(grid_label, obs)
        for seed in range(n_seeds):
            g = make_grid(n, obs, seed=seed)
            start, goal = 0, n * n - 1
            ref = astar(g, start, goal)

            for cfg_name, cfg in ACO_CONFIGS.items():
                if ref is None:
                    rows.append({
                        "instance": inst, "seed": seed, "config": cfg_name,
                        "astar_length": "nan", "astar_turns": "nan",
                        "aco_length": "nan", "aco_turns": "nan", "runtime_ms": "nan",
                    })
                else:
                    t0 = time.perf_counter()
                    path, _ = run_aco(g, start, goal, seed=seed, **ACO_PARAMS, **cfg)
                    ms = (time.perf_counter() - t0) * 1000
                    rows.append({
                        "instance": inst, "seed": seed, "config": cfg_name,
                        "astar_length": round(path_length(ref, n), 4),
                        "astar_turns": path_turns(ref, n),
                        "aco_length": round(path_length(path, n), 4),
                        "aco_turns": path_turns(path, n),
                        "runtime_ms": round(ms, 1),
                    })
                done += 1
                if done % 60 == 0:
                    print(f"  {done}/{total}  {inst} seed={seed} config={cfg_name}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {output_csv}")


if __name__ == "__main__":
    print("Running GA ablation (10x10 + 20x20 only, full seed count)...")
    cheap_instances = [i for i in INSTANCES if i[1] in (10, 20)]
    run_ga_ablation(instances=cheap_instances)
    print("Running ACO ablation (full matrix)...")
    run_aco_ablation()
    print("Done. Run the 50x50 GA ablation separately, e.g. n_seeds=10,")
    print("if you have the runtime budget for it.")