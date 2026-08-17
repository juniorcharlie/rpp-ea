import csv, os, time
from grid import make_grid, astar, path_length, path_turns
from ga import run_ga
from experiments import INSTANCES, GA_PARAMS, _instance_label

def run_one_density(inst_tuple, seeds, configs, out_csv):
    grid_label, n, obs = inst_tuple
    inst = _instance_label(grid_label, obs)
    gp = GA_PARAMS[n]
    file_exists = os.path.exists(out_csv)
    f = open(out_csv, "a", newline="")
    writer = None
    for seed in seeds:
        g = make_grid(n, obs, seed=seed)
        start, goal = 0, n * n - 1
        ref = astar(g, start, goal)
        for cfg_name, cfg in configs.items():
            t0 = time.perf_counter()
            if ref is None:
                row = {"instance": inst, "seed": seed, "config": cfg_name,
                       "astar_length": "nan", "astar_turns": "nan",
                       "ga_length": "nan", "ga_turns": "nan", "runtime_ms": "nan"}
            else:
                path, _ = run_ga(g, start, goal, seed=seed, pc=0.8, pm=0.05, k=3, **gp, **cfg)
                ms = (time.perf_counter() - t0) * 1000
                row = {"instance": inst, "seed": seed, "config": cfg_name,
                       "astar_length": round(path_length(ref, n), 4),
                       "astar_turns": path_turns(ref, n),
                       "ga_length": round(path_length(path, n), 4),
                       "ga_turns": path_turns(path, n),
                       "runtime_ms": round(ms, 1)}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(f"[{inst}] seed={seed} config={cfg_name} -> {time.perf_counter()-t0:.1f}s", flush=True)
    f.close()

if __name__ == "__main__":
    from ablation import GA_CONFIGS
    big = [i for i in INSTANCES if i[1] == 50]
    for idx, inst in enumerate(big):
        run_one_density(inst, seeds=range(10), configs=GA_CONFIGS,
                         out_csv=f"../datasets/raw/ga_ablation_50x50_part{idx}.csv")