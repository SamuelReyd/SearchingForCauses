import numpy as np, json, os
from tqdm import tqdm

from collections import defaultdict

from general import *
from general import Exhaustivness, Models, AlgoTypes
from benchmark_models import get_bbSMK_SCM, get_SMK_SCM

# === Evaluation token ===
def evaluate_smallest(causes, actual_values):
    if not len(causes): return {"accuracy": 0}
    min_pred = min(map(len,causes))
    target_size = int(actual_values["SD"]) + int(actual_values["DK"])
    return {"accuracy": int(min_pred == target_size)}

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
    
    target = {tuple(sorted(cause)) for cause in ref_causes}
    pred = {tuple(sorted(cause)) for cause in causes}
    
    return {
        "Accuracy": int(set(causes) == set(ref_causes)),
        "Recall": r,
        "Precision": p,
        "jaccard": len(target & pred) / len(target | pred) if len(target | pred) else 1,
        "dice": 2 * len(target & pred) / (len(target) + len(pred)) if len(target) + len(pred) else 1,
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


def evaluate_SMK(exh, model, algo, beam_sizes, n_attackers, heuristics, lucb_label, max_steps, folder="results/"):
    file_name = get_file_name(exh, model, algo, heuristics, lucb_label)
    if not os.path.isfile(folder+file_name): 
        print(f"Could not evaluation file {file_name}")
        return
    data = load_json(folder+file_name)
    if exh == Exhaustivness.FULL:
        ref_data = load_json(folder+"base-exact/structured.json")
        ref_causes = get_exact_causes(ref_data)
    else:
        ref_data = None
        ref_causes = None
    for datum in data:
        if datum["beam_size"] == -1: continue
        n = datum["n_attacker"]
        if datum["beam_size"] == -1: continue
        for res in datum["results"]:
            pred = {tuple(cause) for cause in res["causes"]}
            if exh == Exhaustivness.SMALLEST:
                scm = get_SMK_SCM(n, res["context"])
                measures = evaluate_smallest(pred, dict(zip(scm.V,scm.v)))
            else:
                context_repr = int("".join(map(str,res["context"])), 2)
                ref = ref_causes[f"{context_repr}-{n}"]
                measures = evaluate_full(pred, ref)
            res["metrics"] = measures
    
    save_json(folder+file_name, data)

def evaluate_ILP(folder="results/"):
    data = load_json(folder+"base-smallest/ILP.json")
    for datum in data:
        for res in datum["results"]:
            res["metrics"] = {"accuracy": 1.0}
    save_json(folder+"base-smallest/ILP.json", data)
