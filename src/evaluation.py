import numpy as np, json, os
from tqdm import tqdm

from collections import defaultdict

from binary_models import *
from benchmark_models import SMK_model, get_SMK_dim_labels, get_bbSMK_SCM
from benchmark_models import get_noisy_suzy_SCM, get_nSMK_SCM
from actualcauses import beam_search, iterative_identification

def filter_minimality(candidates):
    Cs = [values[0] for values in candidates]
    candidates = [values for values in candidates if not any([set(c) < set(values[0]) for c in Cs])]

    Cs = defaultdict(lambda: [])
    for values in candidates:
        Cs[tuple(values[0])].append(values)
    causes = []
    for cands in Cs.values():
        best = min(cands, key=lambda rule_values: (len(rule_values[1]),rule_values[2]))
        causes.append(best)
    return causes

def build_ref_causes(data):
    ref_causes = defaultdict(lambda: [])
    for datum in data:
        n_attacker = datum["n_attacker"]
        for res in datum["results"]:
            context_repr = int("".join(map(str,res["context"])), 2)
            candidates = zip(res["causes"], res["rules"], res["scores"])
            ref_causes[f"{context_repr}-{n_attacker}"] += candidates
    for key, value in ref_causes.items():
        min_candidates = filter_minimality(value)
        min_causes = [elt[0] for elt in min_candidates]
        
        ref_causes[key] = {tuple(c) for c in min_causes}
    return ref_causes

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
            if ref_cause == cause:
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

def get_exact_causes(prefix=""):
    ref_causes = {}
    data = load_json(prefix+"results/base-exact/structured.json")
    for datum in data:
        n_attacker = datum["n_attacker"]
        for res in datum["results"]:
            context_repr = int("".join(map(str,res["context"])), 2)
            ref_causes[f"{context_repr}-{n_attacker}"] = {tuple(c) for c in res["causes"]}
    return ref_causes

def build_ref_causes_smallest(prefix=""):
    ref_causes = {}
    data = load_json(prefix+"results/base-smallest/ILP.json")
    for datum in data:
        n_attacker = datum["n_attacker"]
        for res in datum["results"]:
            context_repr = int("".join(map(str,res["context"])), 2)
            ref_causes[f"{context_repr}-{n_attacker}"] = set([tuple(res["causes"])])
    return ref_causes

def build_ref_causes_bb(data):
    ref_causes = defaultdict(lambda: set())
    for datum in data:
        n_attacker = datum["n_attacker"]
        for res in datum["results"]:
            context_repr = int("".join(map(str,res["context"])), 2)
            Cs = res["causes"]
            
            SCM = get_bbSMK_SCM(n_attacker, res["context"])
            for C in Cs:
                C_SCM = {}
                for key, value in SCM.items():
                    if key == "simulation":
                        C_SCM[key] = value
                    else:
                        C_SCM[key] = tuple([value[var] for var in C])
                output = beam_search(**C_SCM,
                                     max_steps=-1,beam_size=-1,
                                     early_stop=False, var_mapping=C, verbose=0)
                min_Cs = []
                for values in output:
                    min_Cs.append([C[dim] for dim in values[3]])
                min_Cs={tuple(cause) for cause in min_Cs}
            # if f"{context_repr}-{n_attacker}" == '2638-2': print(min_Cs)
                ref_causes[f"{context_repr}-{n_attacker}"] |= min_Cs
    return ref_causes


def evaluate_SMK(model: Models, exh: Exhaustivness,
                 prefix=""):

    data_struct = None
    if model != Models.BLACK_BOX:
        data_struct = load_json(prefix+f"results/{model.value}-{exh.value}/{AlgoTypes.STRUCTURED.value}.json")
    data_base = load_json(prefix+f"results/{model.value}-{exh.value}/{AlgoTypes.BASE.value}.json")
    if exh == Exhaustivness.SMALLEST:
        ref_causes = build_ref_causes_smallest(prefix)
    elif model == Models.BASE:
        ref_causes = get_exact_causes(prefix)
    elif model == Models.BLACK_BOX:
        ref_causes = build_ref_causes_bb(data_base)
    else:
        ref_causes = build_ref_causes(data_base + data_struct)

    for algo, data in ((AlgoTypes.BASE, data_base),(AlgoTypes.STRUCTURED, data_struct)):
        if data is None: continue
        for datum in tqdm(data):
            n_attacker, beam_size = datum["n_attacker"], datum["beam_size"]
            
            for res in datum["results"]:
                context_repr = int("".join(map(str,res["context"])), 2)
                ref = ref_causes[f"{context_repr}-{n_attacker}"]
                pred = {tuple(cause) for cause in res["causes"]}
                evaluator = evaluators[exh]
                measures = evaluator(pred, ref)
                res |= measures
                metrics = list(measures.keys())
            for m in metrics + ["time"]:
                datum[m+"-avg"] = float(np.mean([res[m] for res in datum["results"]]))
                datum[m+"-std"] = float(np.std([res[m] for res in datum["results"]]))
        
        
        save_json(prefix+f"results/{model.value}-{exh.value}/{algo.value}.json", data)

def evaluate_noisy_SMK(prefix=""):
    # Load Refs
    exh = Exhaustivness.FULL
    data_ref = []
    for algo in AlgoTypes:
        data_ref += load_json(prefix+f"results/{Models.BASE.value}-{exh.value}/{algo.value}.json")
    ref_causes = get_exact_causes(prefix)

    for algo in AlgoTypes:
        for lucb_label in ('lucb','naive'):
            file_name = prefix+f"results/{Models.NOISY.value}-{exh.value}/{algo.value}-{lucb_label}.json"
            if not os.path.isfile(file_name): continue
            data = load_json(file_name)
            for datum in tqdm(data):
                n_attacker, beam_size = datum["n_attacker"], datum["beam_size"]
                
                for res in datum["results"]:
                    context_repr = int("".join(map(str,res["context"])), 2)
                    ref = ref_causes[f"{context_repr}-{n_attacker}"]
                    pred = {tuple(cause) for cause in res["causes"]}
                    evaluator = evaluators[exh]
                    measures = evaluator(pred, ref)
                    res |= measures
                    metrics = list(measures.keys())
                for m in metrics + ["time"]:
                    datum[m+"-avg"] = float(np.mean([res[m] for res in datum["results"]]))
                    datum[m+"-std"] = float(np.std([res[m] for res in datum["results"]]))
        
        
            save_json(file_name, data)

def compute_ref_causes(u):
    n_attacker = len(u)//6
    variables = get_SMK_dim_labels(n_attacker)
    
    SCM = make_SCM(variables, u, SMK_model, n_attacker=n_attacker)
    dag, init_var_ids = build_DAG(n_attacker, SCM["variables"])

    return iterative_identification(**SCM, 
                                     dag=dag, 
                                     init_var_ids=init_var_ids,
                                     max_steps=-1, beam_size=-1, 
                                     verbose=0, early_stop=False)

def evaluate_params_SMK(algo,
                        u=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                        prefix=""):
    ref_causes = compute_ref_causes(u)
    exh = Exhaustivness.FULL
    model = Models.NOISY 
    data = load_json(prefix+f"results/noisy-params/{algo.value}.json")
    for datum in tqdm(data):
        for res in datum["results"]:
            ref = {tuple(cause[3]) for cause in ref_causes}
            pred = {tuple(cause) for cause in res["causes"]}
            evaluator = evaluators[exh]
            measures = evaluator(pred, ref)
            res |= measures
            metrics = list(measures.keys())
        for m in metrics + ["time"]:
            datum[m+"-avg"] = float(np.mean([res[m] for res in datum["results"]]))
            datum[m+"-std"] = float(np.std([res[m] for res in datum["results"]]))
    save_json(prefix+f"results/noisy-params/{algo.value}.json", data)

def evaluate_ILP(prefix):
    data = load_json(prefix+"results/base-smallest/ILP.json")
    for datum in data:
        datum["time-avg"] = float(np.mean([res["time"] for res in datum["results"]]))
        datum["time-std"] = float(np.std([res["time"] for res in datum["results"]]))
        datum["accuracy-avg"] = 1.0
        datum["accuracy-std"] = 0.0
    save_json(prefix+"results/base-smallest/ILP.json", data)