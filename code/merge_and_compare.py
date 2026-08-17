from experiments import merge_results
from stats import comparison_table

merge_results(csv1="../datasets/raw/ga_results_v2.csv",
               csv2="../datasets/raw/pso_aco_results.csv",
               out="../datasets/raw/merged_results_v2.csv")
comparison_table(results_csv="../datasets/raw/merged_results_v2.csv",
                  out_csv="../datasets/raw/comparison_table_v2.csv")