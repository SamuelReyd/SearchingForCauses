import numpy as np, json, os
from tqdm import tqdm

from collections import defaultdict

from binary_models import *
from general import Exhaustivness, Models, AlgoTypes
from benchmark_models import SMK_model, get_SMK_V, get_bbSMK_SCM
from benchmark_models import get_noisy_suzzy_SCM, get_avg_nSMK_SCM, get_lucb_nSMK_SCM
# from actualcauses import beam_search, iterative_identification
from actualcauses_local.base_algorithm import beam_search, get_sets
from actualcauses_local.iterative_subinstance_algorithm import iterative_identification

# === Evaluation token ===
def evaluate_smallest(causes, ref_causes):
    if not len(causes): return {"accuracy": 0}
    if not len(ref_causes): return {"accuracy": -1}
    min_pred = min(map(len,causes))
    min_ref = min(map(len,ref_causes))
    return {"accuracy": int(min_pred == min_ref)}

def evaluate_full(causes, ref_causes):
    n_minimal = 0
    n_non_minimal = 0
    n_missed = 0
    avg_overshot = 0
    for ref_cause in ref_causes:
        for cause in causes:
            if set(ref_cause) < set(cause):
                n_non_minimal += 1
                avg_overshot += len(cause) - len(ref_cause)
                break
            if set(ref_cause) == set(cause):
                n_minimal += 1
                break
        else:
            n_missed += 1
    
    p = n_minimal / len(causes) if len(causes) else 0
    r = n_minimal / len(ref_causes) if len(ref_causes) else 1
    
    return {
        "Accuracy": int(set(causes) == set(ref_causes)),
        "Recall": r,
        "Precision": p,
        "F1": 2 * p * r / (p + r) if p+r else 0, 
        "Missed": n_missed / len(ref_causes) if ref_causes else 1,
        "% Overshoot": n_non_minimal / len(ref_causes) if ref_causes else 1,
        "Average Overshoot": avg_overshot / n_non_minimal if n_non_minimal else 0,
        "Average Repeat": (
            (len(causes) - n_minimal) / n_non_minimal if 
            n_non_minimal 
            else 0
            )
     }

evaluators = {Exhaustivness.FULL: evaluate_full, Exhaustivness.SMALLEST: evaluate_smallest}

# === Reference causes ===
def get_exact_causes(data):
    ref_causes = {}
    for datum in data:
        n_attacker = datum["n_attacker"]
        for res in datum["results"]:
            context_repr = int("".join(map(str,res["context"])), 2)
            ref_causes[f"{context_repr}-{n_attacker}"] = {tuple(sorted(c)) for c in res["causes"]}
    return ref_causes

def build_ref_causes_bb(data):
    ref_causes = defaultdict(lambda: set())
    for datum in data:
        n_attacker = datum["n_attacker"]
        for res in datum["results"]:
            context_repr = int("".join(map(str,res["context"])), 2)
            E = res["rules"]
            scm = get_bbSMK_SCM(n_attacker, res["context"])
            actual_values = dict(zip(scm.V, scm.v))
            for e in E:
                C, W = get_sets(e, actual_values)
                R = tuple([(v,actual_values[v]) for v in W])
                scm.find_causes(I=C, max_steps=-1, beam_size=-1, R=R)
                min_Cs = set(scm.causes_hashable)
                ref_causes[f"{context_repr}-{n_attacker}"] |= min_Cs
    return ref_causes

# === General evaluations ===
def evaluate_SMK(exh, model, algo, beam_sizes, n_attackers, heuristics, lucb_label, max_steps, folder="results/"):
    file_name = get_file_name(exh, model, algo, heuristics, lucb_label)
    if not os.path.isfile(folder+file_name): 
        print(f"Could not evaluation file {file_name}")
        return 
    data = load_json(folder+file_name)
        
    if model == Models.BLACK_BOX:
        ref_causes = build_ref_causes_bb(data)
    else:
        if exh == Exhaustivness.SMALLEST:
            ref_data = load_json(folder+"base-smallest/ILP.json")
        else: 
            ref_data = load_json(folder+"base-exact/structured.json")
        ref_causes = get_exact_causes(ref_data)
    for datum in data:
        n = datum["n_attacker"]
        for res in datum["results"]:
            context_repr = int("".join(map(str,res["context"])), 2)
            ref = ref_causes[f"{context_repr}-{n}"]
            pred = {tuple(cause) for cause in res["causes"]}
            evaluator = evaluators[exh]
            measures = evaluator(pred, ref)
            res["metrics"] = measures
    
    save_json(folder+file_name, data)

def evaluate_ILP(folder="results/"):
    data = load_json(folder+"base-smallest/ILP.json")
    for datum in data:
        for res in datum["results"]:
            res["metrics"] = {"accuracy": 1.0}
    save_json(folder+"base-smallest/ILP.json", data)