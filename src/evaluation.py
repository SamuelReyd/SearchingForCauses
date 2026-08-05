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
            if ref_cause < cause:
                n_non_minimal += 1
                avg_overshot += len(cause) - len(ref_cause)
                break
            if ref_cause == cause:
                n_minimal += 1
                break
        else:
            n_missed += 1
    
    p = n_minimal / len(causes) if len(causes) else 0
    r = n_minimal / len(ref_causes) if len(ref_causes) else 1
    
    target = {tuple(sorted(cause)) for cause in ref_causes}
    pred = {tuple(sorted(cause)) for cause in causes}
    
    return {
        "Accuracy": int(causes == ref_causes),
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

def get_ks(u, is_v=False):
    if not is_v:
        n = len(u) // 6
        scm = get_SMK_SCM(n, u)
        v = scm.v
    else:
        n = (len(v) - 3) // 11
        v = u

    V = get_SMK_V(n)
    k1 = None
    k2 = None
    for label, value in zip(V, v):
        if "DK" in label and label != "DK" and value == 1:
            k1 = int(label.split("-")[1][1:])
        if "SD" in label and label != "SD" and value == 1:
            k2 = int(label.split("-")[1][1:])
    return k1, k2, dict(zip(V,v))

def smk_causes(u, is_v=False):
    k1, k2, v = get_ks(u, is_v)

    if k1:
        dk_causes = [{"DK"},{f"DK-U{k1}"},{f"GP-U{k1}"},{f"GK-U{k1}"}]
        dk_causes.append({variable for variable in (f"FS-U{k1}", f"FN-U{k1}") if v[variable]})
        dk_causes.append({variable for variable in (f"FF-U{k1}", f"FDB-U{k1}") if v[variable]})
    else:
        dk_causes = []

    sd_causes = [
        {"SD"},{f"SD-U{k2}"},{f"KMS-U{k2}"},{f"A-U{k2}"},{f"AD-U{k2}"},
    ] if k2 is not None else []

    if k1 is not None and k2 is not None:
        return [d | s for d in dk_causes for s in sd_causes]

    return dk_causes or sd_causes


def evaluate_SMK(exh, model, algo, beam_sizes, n_attackers, heuristics, lucb_label, max_steps, folder="results/"):
    file_name = get_file_name(exh, model, algo, heuristics, lucb_label)
    if not os.path.isfile(folder+file_name): 
        print(f"Could not evaluation file {file_name}")
        return
    data = load_json(folder+file_name)
    # if exh == Exhaustivness.FULL:
    #     ref_data = load_json(folder+"base-exact/structured.json")
    #     ref_causes = get_exact_causes(ref_data)
    # else:
    #     ref_data = None
    #     ref_causes = None
    for datum in data:
        # if datum["beam_size"] == -1: continue
        n = datum["n_attacker"]
        # if datum["beam_size"] == -1: continue
        for res in datum["results"]:
            # pred = {tuple(cause) for cause in res["causes"]}
            if exh == Exhaustivness.SMALLEST:
                scm = get_SMK_SCM(n, res["context"])
                # measures = evaluate_smallest(pred, dict(zip(scm.V,scm.v)))
                measures = evaluate_smallest(list(map(set,res["causes"])), dict(zip(scm.V,scm.v)))
            else:
                # context_repr = int("".join(map(str,res["context"])), 2)
                # ref = ref_causes[f"{context_repr}-{n}"]
                # measures = evaluate_full(pred, ref)
                measures = evaluate_full(list(map(set,res["causes"])), smk_causes(res["context"]))
            res["metrics"] = measures
    
    save_json(folder+file_name, data)

def evaluate_ILP(folder="results/"):
    data = load_json(folder+"base-smallest/ILP.json")
    for datum in data:
        for res in datum["results"]:
            res["metrics"] = {"accuracy": 1.0}
    save_json(folder+"base-smallest/ILP.json", data)
