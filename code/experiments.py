"""Experiment runners for all three algorithms (GA, PSO, ACO) across the
standard 9-instance x 30-seed benchmark matrix defined in the project
framework (rpp_ea_framework.md, section 5).
"""
import csv
import time
import os

from grid import make_grid, astar, path_length, path_turns
from ga import run_ga
from pso import run_pso
from aco import run_aco

# Experiment matrix (grid label, N, obstacle rate) — matches rpp_ea_framework.md §5
INSTANCES = [
    ("10x10", 10, 0.10), ("10x10", 10, 0.20), ("10x10", 10, 0.30),
    ("20x20", 20, 0.10), ("20x20", 20, 0.20), ("20x20", 20, 0.40),
    ("50x50", 50, 0.10), ("50x50", 50, 0.30), ("50x50", 50, 0.50),
]
N_SEEDS = 30

# GA hyperparameters by grid size (Phase 1 spec, scaled per framework §8)
GA_PARAMS = {
    10: dict(pop_size=50,  n_gen=300),
    20: dict(pop_size=80,  n_gen=500),
    50: dict(pop_size=120, n_gen=800),
}

# PSO / ACO hyperparameters (Phase 2 spec)
PSO_PARAMS = dict(n_particles=80, n_iter=200, w=0.5, c1=1.5, c2=1.5)
ACO_PARAMS = dict(n_ants=50,    n_iter=200, alpha=1.0, beta=2.0, rho=0.1, Q=1.0)


def _instance_label(grid_label, obs_rate):
    return f"{grid_label}_{int(obs_rate*100)}pct"


def run_ga_all(output_csv="../datasets/raw/ga_results.csv"):
    """Run GA on all instances x seeds. Phase 1.
    Writes instance,seed,astar_length,astar_turns,ga_length,ga_turns,runtime_ms
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    rows = []
    total = len(INSTANCES) * N_SEEDS
    done = 0

    for grid_label, n, obs in INSTANCES:
        inst = _instance_label(grid_label, obs)
        gp = GA_PARAMS[n]
        for seed in range(N_SEEDS):
            g = make_grid(n, obs, seed=seed)
            start, goal = 0, n * n - 1
            ref = astar(g, start, goal)

            if ref is None:
                rows.append({
                    "instance": inst, "seed": seed,
                    "astar_length": "nan", "astar_turns": "nan",
                    "ga_length": "nan", "ga_turns": "nan", "runtime_ms": "nan",
                })
            else:
                t0 = time.perf_counter()
                ga_path, _ = run_ga(g, start, goal, seed=seed, pc=0.8, pm=0.05, k=3, **gp)
                ga_ms = (time.perf_counter() - t0) * 1000

                rows.append({
                    "instance": inst, "seed": seed,
                    "astar_length": round(path_length(ref, n), 4),
                    "astar_turns": path_turns(ref, n),
                    "ga_length": round(path_length(ga_path, n), 4),
                    "ga_turns": path_turns(ga_path, n),
                    "runtime_ms": round(ga_ms, 1),
                })

            done += 1
            if done % 30 == 0:
                print(f"  {done}/{total}  {inst} seed={seed}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {output_csv}")


def run_pso_aco(output_csv="../datasets/raw/pso_aco_results.csv"):
    """Run PSO and ACO on all instances x seeds.
    Writes nan for unreachable seeds (matching the GA runner's row format).
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    rows = []
    total = len(INSTANCES) * N_SEEDS
    done = 0

    for grid_label, n, obs in INSTANCES:
        inst = _instance_label(grid_label, obs)
        for seed in range(N_SEEDS):
            g = make_grid(n, obs, seed=seed)
            start, goal = 0, n * n - 1
            ref = astar(g, start, goal)

            if ref is None:
                rows.append({
                    "instance":       inst,
                    "seed":           seed,
                    "astar_length":   "nan",
                    "astar_turns":    "nan",
                    "pso_length":     "nan",
                    "pso_turns":      "nan",
                    "pso_runtime_ms": "nan",
                    "aco_length":     "nan",
                    "aco_turns":      "nan",
                    "aco_runtime_ms": "nan",
                })
            else:
                t0 = time.perf_counter()
                pso_path, _ = run_pso(g, start, goal, seed=seed, **PSO_PARAMS)
                pso_ms = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                aco_path, _ = run_aco(g, start, goal, seed=seed, **ACO_PARAMS)
                aco_ms = (time.perf_counter() - t0) * 1000

                rows.append({
                    "instance":       inst,
                    "seed":           seed,
                    "astar_length":   round(path_length(ref,       n), 4),
                    "astar_turns":    path_turns(ref,       n),
                    "pso_length":     round(path_length(pso_path,  n), 4),
                    "pso_turns":      path_turns(pso_path,  n),
                    "pso_runtime_ms": round(pso_ms, 1),
                    "aco_length":     round(path_length(aco_path,  n), 4),
                    "aco_turns":      path_turns(aco_path,  n),
                    "aco_runtime_ms": round(aco_ms, 1),
                })

            done += 1
            if done % 30 == 0:
                print(f"  {done}/{total}  {inst} seed={seed}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {output_csv}")


def merge_results(csv1="../datasets/raw/ga_results.csv",
                   csv2="../datasets/raw/pso_aco_results.csv",
                   out="../datasets/raw/merged_results.csv"):
    """Join GA results and PSO+ACO results on (instance, seed)."""
    def load(path):
        with open(path, newline="") as f:
            return {(r["instance"], r["seed"]): r for r in csv.DictReader(f)}

    ga_rows  = load(csv1)
    pso_rows = load(csv2)

    merged = []
    for key, ga in ga_rows.items():
        pso = pso_rows.get(key, {})
        merged.append({
            "instance":       ga["instance"],
            "seed":           ga["seed"],
            "astar_length":   ga["astar_length"],
            "astar_turns":    ga["astar_turns"],
            "ga_length":      ga.get("ga_length", "nan"),
            "ga_turns":       ga.get("ga_turns",  "nan"),
            "ga_runtime_ms":  ga.get("runtime_ms", ga.get("ga_runtime_ms", "nan")),
            "pso_length":     pso.get("pso_length",     "nan"),
            "pso_turns":      pso.get("pso_turns",      "nan"),
            "pso_runtime_ms": pso.get("pso_runtime_ms", "nan"),
            "aco_length":     pso.get("aco_length",     "nan"),
            "aco_turns":      pso.get("aco_turns",      "nan"),
            "aco_runtime_ms": pso.get("aco_runtime_ms", "nan"),
        })

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged[0].keys())
        writer.writeheader()
        writer.writerows(merged)
    print(f"Merged {len(merged)} rows → {out}")


if __name__ == "__main__":
    print("Running GA experiments…")
    run_ga_all()
    print("Running PSO+ACO experiments…")
    run_pso_aco()
    merge_results()
    print("Done.")
