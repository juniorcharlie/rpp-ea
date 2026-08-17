"""Statistical comparison: Wilcoxon signed-rank test + summary table.
No scipy — implemented from scratch with stdlib + numpy.
Wilcoxon is run on TURNS (not length) because all EAs match A* on length
(A* is optimal), so length differences are all zero and the test is uninformative.
"""
import csv
import math
import os
import numpy as np
from collections import defaultdict


def wilcoxon_signed_rank(x, y):
    """
    Non-parametric paired test. Returns (W_statistic, p_value).
    Uses normal approximation (valid for n > 5).
    Returns (nan, nan) if fewer than 5 non-zero differences.
    """
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    diffs = x - y
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n < 5:
        return float("nan"), float("nan")

    abs_d = np.abs(diffs)
    order = np.argsort(abs_d)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    # Tie correction
    unique, counts = np.unique(abs_d, return_counts=True)
    for val, cnt in zip(unique, counts):
        if cnt > 1:
            ranks[abs_d == val] = ranks[abs_d == val].mean()

    W = float(ranks[diffs > 0].sum())
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sigma if sigma > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return W, p


def comparison_table(results_csv="results/results_all.csv",
                     out_csv="results/comparison_table.csv"):
    """
    Load merged CSV. Print mean±std for length and turns, plus
    Wilcoxon p-values on TURNS (vs A*) for GA, PSO, ACO.
    """
    raw_len   = defaultdict(lambda: defaultdict(dict))  # [inst][algo][seed] = float
    raw_turns = defaultdict(lambda: defaultdict(dict))

    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = row["instance"]
            seed = row["seed"]
            for algo in ["astar", "ga", "pso", "aco"]:
                for metric, store in [("length", raw_len), ("turns", raw_turns)]:
                    col = f"{algo}_{metric}"
                    if col in row and row[col].strip() and row[col] != "nan":
                        try:
                            store[inst][algo][seed] = float(row[col])
                        except ValueError:
                            pass

    algos  = ["astar", "ga", "pso", "aco"]

    header = (f"{'Instance':<20} "
              + " ".join(f"{'  '+a.upper()+' len':>16}" for a in algos)
              + "  | p(GA)  p(PSO)  p(ACO)  [vs A* TURNS, Wilcoxon]")
    print(header)
    print("-" * len(header))

    table_rows = []
    for inst in sorted(raw_len.keys()):
        # Length stats
        means_l, stds_l = {}, {}
        for a in algos:
            vals = list(raw_len[inst][a].values())
            means_l[a] = np.mean(vals) if vals else float("nan")
            stds_l[a]  = np.std(vals)  if vals else float("nan")

        # Turns stats
        means_t, stds_t = {}, {}
        for a in algos:
            vals = list(raw_turns[inst][a].values())
            means_t[a] = np.mean(vals) if vals else float("nan")
            stds_t[a]  = np.std(vals)  if vals else float("nan")

        # Wilcoxon on turns, paired by seed
        ps = {}
        for a in ["ga", "pso", "aco"]:
            common = set(raw_turns[inst][a].keys()) & set(raw_turns[inst]["astar"].keys())
            if len(common) >= 5:
                x = [raw_turns[inst][a][s]      for s in sorted(common)]
                y = [raw_turns[inst]["astar"][s] for s in sorted(common)]
                _, p = wilcoxon_signed_rank(x, y)
            else:
                p = float("nan")
            ps[a] = p

        def fmtl(m, s):
            if math.isnan(m): return f"{'nan±nan':>16}"
            return f"{m:>8.2f}±{s:<6.2f}"

        def fmtp(p):
            return f"{p:.3f}" if not math.isnan(p) else "  nan"

        row_str = (f"{inst:<20} "
                   + " ".join(fmtl(means_l[a], stds_l[a]) for a in algos)
                   + f"  | {fmtp(ps['ga'])}  {fmtp(ps['pso'])}  {fmtp(ps['aco'])}")
        print(row_str)

        table_rows.append({
            "instance":          inst,
            **{f"{a}_length_mean": f"{means_l[a]:.4f}" for a in algos},
            **{f"{a}_length_std":  f"{stds_l[a]:.4f}"  for a in algos},
            **{f"{a}_turns_mean":  f"{means_t[a]:.4f}" for a in algos},
            **{f"{a}_turns_std":   f"{stds_t[a]:.4f}"  for a in algos},
            "p_ga_vs_astar_turns":  f"{ps['ga']:.4f}"  if not math.isnan(ps["ga"])  else "nan",
            "p_pso_vs_astar_turns": f"{ps['pso']:.4f}" if not math.isnan(ps["pso"]) else "nan",
            "p_aco_vs_astar_turns": f"{ps['aco']:.4f}" if not math.isnan(ps["aco"]) else "nan",
        })

    if table_rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=table_rows[0].keys())
            writer.writeheader()
            writer.writerows(table_rows)
        print(f"\nSaved → {out_csv}")

def ablation_comparison_table(results_csv, metric_col, baseline_config, out_csv=None):
    """
    Pairwise comparison of each ablation config against `baseline_config` on
    `metric_col` (e.g. 'ga_turns' or 'aco_turns'), paired by seed within each
    instance. Same Wilcoxon machinery as comparison_table, but compares
    configs against each other (ablation.py's output) instead of an
    algorithm against A*.
    """
    raw = defaultdict(lambda: defaultdict(dict))

    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(metric_col, "").strip()
            if val and val != "nan":
                try:
                    raw[row["instance"]][row["config"]][row["seed"]] = float(val)
                except ValueError:
                    pass

    table_rows = []
    for inst in sorted(raw.keys()):
        base_vals = raw[inst].get(baseline_config, {})
        row_out = {
            "instance": inst,
            f"{baseline_config}_mean": np.mean(list(base_vals.values())) if base_vals else float("nan"),
        }
        for cfg in [c for c in raw[inst] if c != baseline_config]:
            cfg_vals = raw[inst][cfg]
            common = set(cfg_vals) & set(base_vals)
            row_out[f"{cfg}_mean"] = np.mean(list(cfg_vals.values())) if cfg_vals else float("nan")
            if len(common) >= 5:
                x = [cfg_vals[s] for s in sorted(common)]
                y = [base_vals[s] for s in sorted(common)]
                _, p = wilcoxon_signed_rank(x, y)
            else:
                p = float("nan")
            row_out[f"p_{cfg}_vs_{baseline_config}"] = p
        table_rows.append(row_out)
        print(inst, {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in row_out.items() if k != "instance"})

    if out_csv and table_rows:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=table_rows[0].keys())
            writer.writeheader()
            writer.writerows(table_rows)
        print(f"Saved -> {out_csv}")

    return table_rows

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.integers(0, 10, 30).tolist()
    y = rng.integers(2, 12, 30).tolist()
    W, p = wilcoxon_signed_rank(x, y)
    print(f"Wilcoxon smoke test: W={W:.1f} p={p:.4f}")
    assert 0.0 <= p <= 1.0
    print("stats.py self-check passed")
