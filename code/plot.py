"""Plotting utilities for RPP-EA Phase 1 + Phase 2."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# Colour scheme (spec)
COLORS = {
    "A*":  "steelblue",
    "GA":  "coral",
    "PSO": "mediumseagreen",
    "ACO": "mediumpurple",
}

# ── Phase 1 plots (kept intact) ─────────────────────────────────────────────

def grid_viz(grid, path, title, save_path):
    """Visualise grid with path overlaid."""
    import numpy as np
    n = grid.shape[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid, cmap="binary", origin="upper")
    if path:
        rs = [c // n for c in path]
        cs = [c %  n for c in path]
        ax.plot(cs, rs, "b-", lw=1.5)
        turns = [path[i] for i in range(1, len(path)-1)
                 if (path[i]//n - path[i-1]//n, path[i]%n - path[i-1]%n) !=
                    (path[i+1]//n - path[i]//n, path[i+1]%n - path[i]%n)]
        if turns:
            ax.plot([c%n for c in turns], [c//n for c in turns], "ro", ms=4)
        ax.plot(path[0]%n,  path[0]//n,  "gs", ms=8, label="Start")
        ax.plot(path[-1]%n, path[-1]//n, "rs", ms=8, label="Goal")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def boxplot_comparison(results_csv, metric, ylabel, title, save_path):
    """Phase 1: A* vs GA boxplot per grid size."""
    data = defaultdict(lambda: defaultdict(list))
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            parts = row["instance"].split("_")
            grid, obs = parts[0], parts[1]
            a_val, g_val = row[f"astar_{metric}"], row[f"ga_{metric}"]
            if a_val != "nan" and g_val != "nan":
                data[grid][obs].append((float(a_val), float(g_val)))

    grid_order = ["10x10", "20x20", "50x50"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)
    for ax, grid in zip(axes, grid_order):
        obs_keys = sorted(data[grid].keys())
        positions, labels = [], []
        for i, obs in enumerate(obs_keys):
            pairs = data[grid][obs]
            astar_vals = [p[0] for p in pairs]
            ga_vals    = [p[1] for p in pairs]
            bp = ax.boxplot([astar_vals, ga_vals],
                            positions=[i*3+1, i*3+2], widths=0.7,
                            patch_artist=True,
                            boxprops=dict(facecolor="steelblue"),
                            medianprops=dict(color="darkorange", lw=2))
            bp["boxes"][1].set_facecolor("coral")
            labels.append(obs)
            positions.append(i*3+1.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(grid)
        ax.set_xlabel("Obstacle rate")
        ax.set_ylabel(ylabel)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="steelblue", label="A*"),
                            Patch(color="coral",     label="GA")])
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

# ── Phase 2 plots ────────────────────────────────────────────────────────────

def convergence_plot_multi(histories_dict, label, save_path):
    """
    Multi-algorithm convergence curve.
    histories_dict = {"GA": [...], "PSO": [...], "ACO": [...]}
    x = iteration/generation, y = best fitness.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for algo, hist in histories_dict.items():
        ax.plot(hist, label=algo, color=COLORS.get(algo, "gray"), lw=1.5)
    ax.set_xlabel("Iteration / Generation")
    ax.set_ylabel("Best fitness")
    ax.set_title(f"Convergence — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def boxplot_comparison_all(results_csv, metric, ylabel, title, save_path):
    """Phase 2: A*, GA, PSO, ACO side-by-side per grid size."""
    algos = ["astar", "ga", "pso", "aco"]
    labels = ["A*", "GA", "PSO", "ACO"]
    colors = [COLORS[l] for l in labels]

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            parts = row["instance"].split("_")
            grid, obs = parts[0], parts[1]
            for a in algos:
                col = f"{a}_{metric}"
                if col in row and row[col] and row[col] != "nan":
                    data[grid][obs][a].append(float(row[col]))

    grid_order = ["10x10", "20x20", "50x50"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title)
    n_algos = len(algos)

    for ax, grid in zip(axes, grid_order):
        obs_keys = sorted(data[grid].keys())
        n_obs = len(obs_keys)
        spacing = n_algos + 1  # gap between obs groups

        for i, obs in enumerate(obs_keys):
            for j, (a, lbl, col) in enumerate(zip(algos, labels, colors)):
                vals = data[grid][obs][a]
                if not vals:
                    continue
                pos = i * spacing + j + 1
                bp = ax.boxplot(vals, positions=[pos], widths=0.7,
                                patch_artist=True,
                                boxprops=dict(facecolor=col, alpha=0.8),
                                medianprops=dict(color="black", lw=1.5),
                                whiskerprops=dict(color="gray"),
                                capprops=dict(color="gray"),
                                flierprops=dict(marker="o", ms=3, color="gray"))

        group_centers = [i * spacing + (n_algos / 2 + 0.5) for i in range(n_obs)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(obs_keys)
        ax.set_title(grid)
        ax.set_xlabel("Obstacle rate")
        ax.set_ylabel(ylabel)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)],
                  fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
