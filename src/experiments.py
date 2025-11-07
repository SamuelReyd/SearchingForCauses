import numpy as np, json, os

from ILP_why import run_ilp_SMK
from general import *
from identification import run_ILP_SMK, evaluate_heuristics, run_SMK, run_noisy_SMK
from evaluation import evaluate_SMK, evaluate_noisy_SMK, evaluate_params_SMK


# Define experiments and parameters
exps = (
        (Exhaustivness.FULL, Models.BASE, AlgoTypes.BASE),
        (Exhaustivness.FULL, Models.BASE, AlgoTypes.STRUCTURED),
        (Exhaustivness.FULL, Models.NON_BOOLEAN, AlgoTypes.BASE),
        (Exhaustivness.FULL, Models.NON_BOOLEAN, AlgoTypes.STRUCTURED),
        (Exhaustivness.FULL, Models.BLACK_BOX, AlgoTypes.BASE),
    )

# See hyperparameters in general.py
    
# Main
if __name__ == "__main__":
    # == Contexts ==
    make_base_contexts(N, full_attackers)

    # == ILP ==
    # if not os.path.isfile(f"results/base-smallest/ILP.json"):
    #     print("Run ILP")
    #     run_ILP_SMK(n_attackers_smallest, prefix="") 

    # == Heuristics ==
    # if not os.path.isfile(f"results/base-full/heuristics.json"):
    #     evaluate_heuristics(n_attacker=5, N=50, measure="F1", prefix="")
    
    # == Exact results - base model ==
    if not os.path.isfile(f"results/{Models.BASE.value}-{Exhaustivness.EXACT.value}/{AlgoTypes.STRUCTURED.value}.json"):
        run_SMK(n_attackers, [-1], Exhaustivness.EXACT, Models.BASE, AlgoTypes.STRUCTURED, max_steps=7)
            
    # Main experiments - deterministic
    for exh, model, algo in exps:
        if os.path.isfile(f"results/{model.value}-{exh.value}/{algo.value}.json"): 
            continue
        run_SMK(n_attackers, beam_sizes, exh, model, algo, max_steps=7)
    print("Evaluation...")
    evaluate_SMK(model, exh)
    print()
        
    # Main experiments - noisy
    # exh = Exhaustivness.FULL
    # model = Models.NOISY
    # for algo in AlgoTypes:
    #     for do_lucb in (True, False):
    #         lucb_label = "lucb" if do_lucb else "naive"
    #         if os.path.isfile(f"results/{model.value}-{exh.value}/{algo.value}-{lucb_label}.json"):
    #             continue
    #         run_noisy_SMK(algo, n_attackers, beam_sizes, max_steps, lucb_params, n_seeds, nl, do_lucb)
    # evaluate_noisy_SMK(prefix="")
    
    # Smallest identification
    # for algo in AlgoTypes:
    #     if os.path.isfile(f"results/base-smallest/{algo.value}.json"):
    #         continue
    #     run_SMK(n_attackers_smallest, beam_sizes_smallest, Exhaustivness.SMALLEST, Models.BASE, algo, max_steps=-1)
    #     print()
    # evaluate_SMK(Models.BASE, Exhaustivness.SMALLEST, prefix="")
