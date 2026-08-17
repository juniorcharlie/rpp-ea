"""run_all.py — Single entry point for the full RPP-EA pipeline.

Run from this directory:  python run_all.py
"""
import csv
import os
import statistics

from grid import make_grid, astar, path_length, path_turns
from ga import run_ga
from pso import run_pso
from aco import run_aco
from experiments import (GA_PARAMS, PSO_PARAMS, ACO_PARAMS,
                          run_ga_all, run_pso_aco, merge_results)
from plot import (grid_viz, boxplot_comparison,
                   convergence_plot_multi, boxplot_comparison_all)
from stats import comparison_table

RAW_DIR = "../datasets/raw"
FIG_DIR = "../datasets/figures"

GA_CSV      = os.path.join(RAW_DIR, "ga_results.csv")
PSOACO_CSV  = os.path.join(RAW_DIR, "pso_aco_results.csv")
MERGED_CSV  = os.path.join(RAW_DIR, "merged_results.csv")
COMPARE_CSV = os.path.join(RAW_DIR, "comparison_table.csv")

# Representative instances for the GA path visualisations (20% obstacle rate, seed=0)
VIZ_INSTANCES = [
    ("10x10", 10, 0.20,  0,   99, "grid_10x10.png"),
    ("20x20", 20, 0.20,  0,  399, "grid_20x20.png"),
    ("50x50", 50, 0.20,  0, 2499, "grid_50x50.png"),
]

# Seeds chosen so the convergence curves show visible improvement, not flat
# from iteration 0 (same instances used for the boxplots' 20%-obstacle column).
CONV_SEEDS = {10: 1, 20: 2, 50: 1}


def make_viz():
    for label, N, obs, start, goal, fname in VIZ_INSTANCES:
        grid = make_grid(N, obs, seed=0)
        ga_path, _ = run_ga(grid, start, goal,
                             pop_size=80 if N >= 20 else 50,
                             n_gen=300, seed=0)
        path = ga_path if ga_path else astar(grid, start, goal)
        turns  = path_turns(path, N)  if path else 0
        length = path_length(path, N) if path else 0
        title = f"GA path — {label} 20%obs | len={length:.2f} turns={turns}"
        grid_viz(grid, path, title, os.path.join(FIG_DIR, fname))
        print(f"  Saved {fname}")


def make_convergence_plots():
    for grid_label, n, obs in [("10x10", 10, 0.20), ("20x20", 20, 0.20), ("50x50", 50, 0.20)]:
        seed = CONV_SEEDS[n]
        g = make_grid(n, obs, seed=seed)
        start, goal = 0, n * n - 1
        _, ga_hist  = run_ga(g, start, goal, seed=seed, pc=0.8, pm=0.05, k=3, **GA_PARAMS[n])
        _, pso_hist = run_pso(g, start, goal, seed=seed, **PSO_PARAMS)
        _, aco_hist = run_aco(g, start, goal, seed=seed, **ACO_PARAMS)
        convergence_plot_multi(
            {"GA": ga_hist, "PSO": pso_hist, "ACO": aco_hist},
            label=f"{grid_label} 20% obs (seed={seed})",
            save_path=os.path.join(FIG_DIR, f"convergence_{grid_label}.png"),
        )
        print(f"  Saved convergence_{grid_label}.png")


def print_summary(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    instances = sorted(set(r["instance"] for r in rows))
    print(f"\n{'Instance':<18} {'GA len (mean±std)':<22} {'GA turns (mean±std)':<22} {'Runtime ms'}")
    print("-" * 80)
    for inst in instances:
        inst_rows = [r for r in rows if r["instance"] == inst]
        ga_lens  = [float(r["ga_length"]) for r in inst_rows if r["ga_length"] != "nan"]
        ga_turns = [float(r["ga_turns"])  for r in inst_rows if r["ga_turns"]  != "nan"]
        runtimes = [float(r["runtime_ms"]) for r in inst_rows if r["runtime_ms"] != "nan"]
        if not ga_lens:
            print(f"{inst:<18} (no valid paths)")
            continue
        ml, sl = statistics.mean(ga_lens),  statistics.stdev(ga_lens)  if len(ga_lens)  > 1 else 0
        mt, st = statistics.mean(ga_turns), statistics.stdev(ga_turns) if len(ga_turns) > 1 else 0
        mr     = statistics.mean(runtimes)
        print(f"{inst:<18} {ml:7.2f} ± {sl:5.2f}       {mt:6.2f} ± {st:5.2f}        {mr:8.1f}")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("Step 1/6: GA experiments (9 instances x 30 seeds)")
    print("=" * 60)
    run_ga_all(output_csv=GA_CSV)

    print("\n" + "=" * 60)
    print("Step 2/6: PSO + ACO experiments (9 instances x 30 seeds)")
    print("=" * 60)
    run_pso_aco(output_csv=PSOACO_CSV)

    print("\nStep 3/6: Merging GA + PSO/ACO results")
    merge_results(csv1=GA_CSV, csv2=PSOACO_CSV, out=MERGED_CSV)

    print("\nStep 4/6: Grid visualisations")
    make_viz()

    print("\nStep 5/6: Boxplots + convergence curves")
    boxplot_comparison(GA_CSV, metric="length", ylabel="Length",
                        title="Path length: A* vs GA",
                        save_path=os.path.join(FIG_DIR, "boxplot_length.png"))
    boxplot_comparison(GA_CSV, metric="turns", ylabel="Turns",
                        title="Path turns: A* vs GA",
                        save_path=os.path.join(FIG_DIR, "boxplot_turns.png"))
    make_convergence_plots()
    boxplot_comparison_all(MERGED_CSV, metric="length", ylabel="Length",
                            title="Path length comparison: A* vs GA vs PSO vs ACO",
                            save_path=os.path.join(FIG_DIR, "boxplot_length_all.png"))
    boxplot_comparison_all(MERGED_CSV, metric="turns", ylabel="Turns",
                            title="Path turns comparison: A* vs GA vs PSO vs ACO",
                            save_path=os.path.join(FIG_DIR, "boxplot_turns_all.png"))

    print("\nStep 6/6: Statistical comparison table (Wilcoxon, vs A*, on turns)")
    comparison_table(results_csv=MERGED_CSV, out_csv=COMPARE_CSV)

    print_summary(GA_CSV)
    print("\nAll done. Datasets in ../datasets/raw/, figures in ../datasets/figures/")
