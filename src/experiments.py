import numpy as np, json, os

from general import *
from identification import run_ILP_SMK, run_SMK , heuristics_refs
from evaluation import evaluate_SMK, evaluate_ILP

# === Hyper parameters ===
# == General ==
N = 20
n_attackers = np.arange(2, 15,2).tolist()
ref_attackers = (2,5,10)
beam_sizes = [2,4,8,16,32,64,128,256]

# beam_sizes_smallest = [  1,  11,  22,  33,  44,  55,  66,  77,  88, 100]
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
    # exh, model, algo, beam_sizes, n_attackers, heuristics, lucb_label
    # Exact results
    (Exhaustivness.EXACT, Models.BASE, AlgoTypes.STRUCTURED, [-1], n_attackers, [None], None, -1),
    # Smallest
    (Exhaustivness.SMALLEST, Models.BASE, AlgoTypes.BASE, beam_sizes, n_attackers, [None], None, -1),
    (Exhaustivness.SMALLEST, Models.BASE, AlgoTypes.STRUCTURED, beam_sizes, n_attackers, [None], None, -1),
    # Full Base
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.BASE, beam_sizes, n_attackers, [None], None, 7),
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.STRUCTURED, beam_sizes, n_attackers, [None], None, 7),
    # Full Non-Boolean
    (Exhaustivness.FULL, Models.NON_BOOLEAN, AlgoTypes.BASE, beam_sizes, ref_attackers, [None], None, 7),
    (Exhaustivness.FULL, Models.NON_BOOLEAN, AlgoTypes.STRUCTURED, beam_sizes, ref_attackers, [None], None, 7),
    # Heuristics
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.BASE, [64], [2], heuristics_refs.keys(), None, 7),
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.STRUCTURED, [64], [10], heuristics_refs.keys(), None, 7),
    # Noisy
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.STRUCTURED, beam_sizes, ref_attackers, [None], "naive", 7),
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.STRUCTURED, beam_sizes, ref_attackers, [None], "lucb", 7),
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.BASE, beam_sizes, ref_attackers, [None], "naive", 7),
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.BASE, beam_sizes, ref_attackers, [None], "lucb", 7),
    )
    
# Main
if __name__ == "__main__":
    # == Contexts == 
    make_base_contexts(N, n_attackers)
    
    # == ILP ==
    # if not os.path.isfile(f"results/base-smallest/ILP.json"):
    #     print("Run ILP")
    #     run_ILP_SMK(n_attackers_smallest)
    #     evaluate_ILP()

    # == Experiments ==
    for exp in exps:
        run_SMK(*exp, lucb_params, nl, n_seeds)
        evaluate_SMK(*exp)
