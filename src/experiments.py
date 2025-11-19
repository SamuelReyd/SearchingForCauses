import numpy as np, json, os

from general import *
from identification import run_ILP_SMK, run_SMK , heuristics_refs
from evaluation import evaluate_SMK, evaluate_ILP

# === Hyper parameters ===
# == General ==
N = 20
n_attackers = (2,5,10)
n_attackers_smallest = [2,3,4,5,6,7]
full_attackers = list(set(n_attackers) | set(n_attackers_smallest))
beam_sizes = [ 1, 50, 100, 150, 200, 250, 300]
# beam_sizes = [ 1, 25, 50, 75, 100, 125, 150]
# beam_sizes = [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
beam_sizes = [4,8,16,32,64,128,256,512]

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
               "max_iter": 20, 
               "verbose": 0, 
               "init_batch_size": 30,
               "batch_size": 10,
               "delta": .05
               }

# Define experiments and parameters
exps = (
    # exh, model, algo, beam_sizes, n_attackers, heuristics, lucb_label
    # Smallest
    (Exhaustivness.SMALLEST, Models.BASE, AlgoTypes.BASE, beam_sizes, n_attackers_smallest, [None], None, -1),
    (Exhaustivness.SMALLEST, Models.BASE, AlgoTypes.STRUCTURED, beam_sizes, n_attackers_smallest, [None], None, -1),
    # Full deterministic
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.BASE, beam_sizes, n_attackers, [None], None, 7),
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.STRUCTURED, beam_sizes, n_attackers, [None], None, 7),
    (Exhaustivness.FULL, Models.NON_BOOLEAN, AlgoTypes.BASE, beam_sizes, n_attackers, [None], None, 7),
    (Exhaustivness.FULL, Models.NON_BOOLEAN, AlgoTypes.STRUCTURED, beam_sizes, n_attackers, [None], None, 7),
    (Exhaustivness.FULL, Models.BLACK_BOX, AlgoTypes.BASE, beam_sizes, n_attackers, [None], None, 7),
    # Heuristics
    (Exhaustivness.FULL, Models.BASE, AlgoTypes.STRUCTURED, [50], [5], heuristics_refs.keys(), None, 7),
    # Noisy
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.STRUCTURED, beam_sizes, n_attackers, [None], "naive", 7),
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.STRUCTURED, beam_sizes, n_attackers, [None], "lucb", 7),
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.BASE, beam_sizes, n_attackers, [None], "naive", 7),
    (Exhaustivness.FULL, Models.NOISY, AlgoTypes.BASE, beam_sizes, n_attackers, [None], "lucb", 7),
    # Exact results
    (Exhaustivness.EXACT, Models.BASE, AlgoTypes.STRUCTURED, [-1], n_attackers, [None], None, -1),
    )
    
# Main
if __name__ == "__main__":
    # == Contexts == 
    make_base_contexts(N, full_attackers)
    
    # == ILP ==
    if not os.path.isfile(f"results/base-smallest/ILP.json"):
        print("Run ILP")
        run_ILP_SMK(n_attackers_smallest)
        evaluate_ILP()

    # == Experiments ==
    for exp in exps:
        run_SMK(*exp, lucb_params, nl, n_seeds)
        evaluate_SMK(*exp)
