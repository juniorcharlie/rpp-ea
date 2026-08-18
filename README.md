# RPP-EA: Evolutionary Algorithms for Robot Path Planning

Code accompanying the paper *"A Shared-Codebase Ablation Study of GA, PSO,
and ACO for Grid-Based Robot Path Planning"* (preprint on engrXiv).

This repository contains a benchmark framework comparing three evolutionary
algorithms — **Genetic Algorithm (GA)**, **Particle Swarm Optimisation (PSO)**,
and **Ant Colony Optimisation (ACO)** — against an **A\*** baseline for grid-based
robot path planning. The framework evaluates path length and **turn count**
across a 9-instance × 30-seed benchmark matrix, with statistical comparison
(Wilcoxon signed-rank test) and ablation experiments.

## Repository structure

    rpp-ea/
    ├── code/                  # all experiment code (run everything from here)
    │   ├── grid.py            # grid model, A* baseline, path validation, metrics
    │   ├── ga.py              # Genetic Algorithm for path planning
    │   ├── pso.py             # discrete Particle Swarm Optimisation
    │   ├── aco.py             # Ant Colony Optimisation (grid adaptation)
    │   ├── experiments.py     # runners for the 9-instance x 30-seed matrix
    │   ├── stats.py           # Wilcoxon signed-rank test + summary tables
    │   ├── plot.py            # figures: grid visualisations, boxplots
    │   ├── run_all.py         # single entry point — full pipeline end to end
    │   ├── ablation.py        # ablation experiments (shared-framework components)
    │   ├── run_50x50_ablation.py  # reduced-scope ablation on 50x50 grids
    │   └── merge_and_compare.py   # merge raw CSVs, build comparison table
    ├── test_instances/        # pre-generated benchmark grids (.npy)
    │   ├── 10x10_10pct/ ... 50x50_50pct/   # 9 configurations x 30 seeds
    │   └── manifest.csv       # index of every instance: size, obstacle rate,
    │                          #   seed, cell counts, reachability
    ├── requirements.txt
    ├── LICENSE                # MIT
    └── README.md

## Requirements

- Python 3.10+
- `numpy`, `matplotlib`

```bash
pip install -r requirements.txt
```

## Quick start

Run everything from inside the `code/` directory:

```bash
cd code
python run_all.py
```

Results and figures are written to `datasets/raw/` and `datasets/figures/`
at the repository root (created automatically; not tracked by git).

**Runtime warning:** a single GA run on a 50×50 grid takes ~330 s. The full
benchmark matrix (9 instances × 30 seeds × 3 algorithms) is expensive. Use the
`instances=` / `n_seeds=` overrides in the experiment scripts to scope runs —
e.g. full matrix on 10×10/20×20, reduced-seed pass on 50×50.

## Reproducibility

Benchmark grids are pre-generated and shipped as `.npy` files in
`test_instances/` so that GA, PSO, and ACO can be evaluated on byte-identical
maps, independent of the numpy RNG version. All algorithm runs are seeded.

Statistical comparison uses the Wilcoxon signed-rank test on **turn count**
(not path length): A* is length-optimal, so length differences are all zero
and the test would be uninformative on that metric.

## Citation

If you use this code, please cite:

```bibtex
@misc{charlie2026rppea,
  author = {Junior Charlie},
  title  = {A Shared-Codebase Ablation Study of GA, PSO, and ACO
            for Grid-Based Robot Path Planning},
  year   = {2026},
  howpublished = {engrXiv preprint},
  note   = {\url{https://doi.org/10.31224/7994}}
}
```

## License

MIT — see [LICENSE](LICENSE).
