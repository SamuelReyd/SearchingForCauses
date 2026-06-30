import numpy as np, json, os, shutil

from general import *
from identification import run_ILP_SMK, run_SMK , heuristics_refs
from evaluation import evaluate_SMK, evaluate_ILP

# === Hyper parameters ===
# == General ==
N = 20
reg_attackers = np.arange(2, 15, 2).tolist()
smallest_attackers = np.arange(2, 15).tolist()
n_attackers = [2,5,10]

beam_sizes = [2,4,8,16,32,64,128,256]

max_steps = 7

# == Noisy global params ==
nl = 1.5
n_seeds = 10

# == LUCB params ==
lucb_params = {"a": .65, 
               "cause_eps": .1, 
               "non_cause_eps": .1, 
               "beam_eps": .1, 
               "max_iter": 50, 
               "verbose": 0, 
               "init_batch_size": 30,
               "batch_size": 10,
               "delta": .05
               }

# Define experiments and parameters
exps = (
    # Exhaustivness, Model, AlgoType, beam_size, n_attackers, heuristics, stoch algo, max_steps
    # Exact results
    (Exhaustivness.EXACT, Models.BASE, AlgoTypes.STRUCTURED, [-1], n_attackers, [None], None, -1),
    # Full Base
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.BASE, beam_sizes, n_attackers, [None], None, 7),
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.STRUCTURED, beam_sizes, n_attackers, [None], None, 7),
    # Full Non-Boolean
    (Exhaustivness.FULL, Models.NON_BOOLEAN, AlgoTypes.BASE, beam_sizes, n_attackers, [None], None, 7),
    (Exhaustivness.FULL, Models.NON_BOOLEAN, AlgoTypes.STRUCTURED, beam_sizes, n_attackers, [None], None, 7),
    # Heuristics
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.BASE, [8], [2], heuristics_refs.keys(), None, 7),
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.STRUCTURED, [64], [10], heuristics_refs.keys(), None, 7),
    # Noisy
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.STRUCTURED, beam_sizes, n_attackers, [None], "naive", 7),
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.STRUCTURED, beam_sizes, n_attackers, [None], "lucb", 7),
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.BASE, beam_sizes, n_attackers, [None], "naive", 7),
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.BASE, beam_sizes, n_attackers, [None], "lucb", 7),
    )


exps_reg = (
    # Exhaustivness, Model, AlgoType, beam_size, n_attackers, heuristics, stoch algo, max_steps
    # Full reg
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.BASE, beam_sizes, reg_attackers, [None], None, 7),
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.STRUCTURED, beam_sizes, reg_attackers, [None], None, 7),
    # Smallest
    (Exhaustivness.SMALLEST, Models.BASE, AlgoTypes.BASE, beam_sizes, reg_attackers, [None], None, -1),
    (Exhaustivness.SMALLEST, Models.BASE, AlgoTypes.STRUCTURED, beam_sizes, reg_attackers, [None], None, -1),
)

exps_smallest = (
    (Exhaustivness.SMALLEST, Models.BASE, AlgoTypes.BASE, [4, 32, 256], smallest_attackers, [None], None, 7),
    (Exhaustivness.SMALLEST, Models.BASE, AlgoTypes.STRUCTURED, [4, 32, 256], smallest_attackers, [None], None, 7)
)

# Main
if __name__ == "__main__":
    pass
    # == Contexts == 
    # make_base_contexts(N, reg_attackers + n_attackers)
    # shutil.copytree("results/contexts", "results_reg/", dirs_exist_ok=True)
    
    # == Experiments ==
    # for exp in exps:
    #     run_SMK(*exp, lucb_params, nl, n_seeds)
    #     evaluate_SMK(*exp)
        
    # == Experiments for regressions ==
    # # = ILP =
    # # Make contexts for the smallest cause evaluation
    # make_base_contexts(N, smallest_attackers, "results_smallest/")
    
    # if not os.path.isfile(f"results_smallest/base-smallest/ILP.json"):
    #     print("Run ILP")
    #     run_ILP_SMK(smallest_attackers, "results_smallest/")
    #     evaluate_ILP(folder="results_smallest/")
    # # = Ours =
    # for exp in exp_smallest:
    #     run_SMK(*exp, lucb_params=lucb_params, nl=nl, n_seeds=n_seeds, folder="results_smallest/")
    #     evaluate_SMK(*exp, folder="results_smallest/")

    # == Experiments for regressions
    # for exp in exps_reg:
    #     run_SMK(*exp, lucb_params, nl, n_seeds, folder="results_reg/")