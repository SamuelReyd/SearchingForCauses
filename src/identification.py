import numpy as np, json
from tqdm import tqdm

from itertools import product
from collections import defaultdict

from binary_models import *
from benchmark_models import SMK_model, get_SMK_SCM, get_SMK_V, get_mSMK_SCM, get_bbSMK_SCM, get_avg_nSMK_SCM, get_lucb_nSMK_SCM
# from actualcauses import beam_search, iterative_identification
from actualcauses_local.base_algorithm import beam_search
from actualcauses_local.iterative_subinstance_algorithm import iterative_identification

from ILP_why import run_ilp_SMK
from general import *


heuristics_refs = {
    "Sum-pos": lambda s, v: sum(s) - 1,
    "Sum-dif": lambda s, v: sum([s_cf != s_act for s_cf, s_act in zip(s, v)]),
    "Sum-neg": lambda s, v: len(s) - sum(s) + 1,
    "Occam": lambda s, v: len(s) - sum([s_cf != s_act for s_cf, s_act in zip(s, v)]),
    "Rand": lambda s, v: int(np.random.randint(len(s))),
    "Const": lambda s, v: len(s)//2,
    }


# === General function ===
def run_SMK(exh, model, algo, beam_sizes, n_attackers, heuristics, lucb_label, 
            max_steps, lucb_params, nl, n_seeds, folder="results/"):
    results = []
    file_name = get_file_name(exh, model, algo, heuristics, lucb_label)
    if os.path.isfile(folder+file_name): 
        return 
    print("Indentify for:")
    print(f"  Exhaustiveness: {exh.value}")
    print(f"  Version of SMK: {model.value}")
    print(f"  Algorithm used: {algo.value}")
    print(f"  Stoc algorithm: {lucb_label}")
    print(f"  Max n of steps: {max_steps}")
    print(f"{n_attackers=}, {beam_sizes=}, {heuristics=}")
    for n, bs, heuristic in tqdm(list(product(n_attackers, beam_sizes, heuristics))):
        contexts = np.load(folder+f"contexts/n_attacker={n}.npy")
        data = run_one_SMK(contexts, exh, model, algo, bs, n, heuristic, lucb_label, 
                           max_steps, lucb_params, nl, n_seeds, verbose=0)
        results.append(data)
    save_json(folder+file_name, results)
    
def run_one_SMK(contexts, exh, model, algo, bs, n, heuristic, lucb_label, 
                max_steps, lucb_params, nl, n_seeds, verbose=0):
    data = {
        "exhaustiveness": exh.value,
        "model": model.value,
        "algo": algo.value,
        "beam_size": bs,
        "n_attacker": n,
        "heuristic": heuristic,
        "lucb_label": lucb_label,
        "lucb_params": lucb_params if lucb_label == "lucb" else None,
        "nl": nl if lucb_label else None,
        "max_steps": max_steps,
        "results": []
    }
    assert not (model == Models.BLACK_BOX and algo == AlgoTypes.STRUCTURED)
    assert not (heuristic is not None and model != Models.BASE)
    seeds = [None] if model != Models.NOISY else range(n_seeds)
    for u, seed in tqdm(product(contexts, seeds), disable=not verbose):
        # Adapt on exhaustivness
        early_stop = (exh == Exhaustivness.SMALLEST)
        u = u.tolist()
        # Adapt on the version of the SCM
        if model == Models.BASE: scm = get_SMK_SCM(n, u, heuristics_refs[heuristic])
        elif model == Models.NON_BOOLEAN: scm = get_mSMK_SCM(n, u)
        elif model == Models.BLACK_BOX: scm = get_bbSMK_SCM(n, u)
        elif model == Models.NOISY: 
            params = lucb_params | {"beam_size": bs}
            params["lucb_info"] = defaultdict(lambda: 0)
            if lucb_label == "lucb": scm = get_lucb_nSMK_SCM(n, u, nl, params)
            else: scm = get_avg_nSMK_SCM(n, u, lucb_params["max_iter"], nl)
        else:
            raise Exception(f"Model {model} not handled")
        use_ISI = algo==AlgoTypes.STRUCTURED
        np.random.seed(seed)
        scm.find_causes(ISI=use_ISI, beam_size=bs, epsilon=lucb_params["a"],
                        max_steps=max_steps, early_stop=early_stop)
        res = {
            "rules": serialize_interventions(scm.interventions),
            "causes": scm.causes_hashable,
            "context": u,
            "time": scm.identification_time,
            "seed": seed
        }
        if lucb_label == "lucb": res["lucb_info"] = params["lucb_info"]
        else: res["lucb_info"] = None
        data["results"].append(res)
    return data


# === ILP ===
def serialize_symbols(C, variables):
    var_map = {val: i for i, val in enumerate(variables)}
    return [var_map[s.name] for s in C]

def serialize_symbols_values(C, W, instance, variables):
    var_map = {val: i for i, val in enumerate(variables)}
    rules = []
    for s in C:
        var_id = var_map[s.name]
        rules.append((var_id,1-int(instance[var_id])))
    for s in W:
        var_id = var_map[s.name]
        rules.append((var_id,int(instance[var_id])))
    return rules

def run_ILP_SMK(n_attackers, prefix=""):
    model = Models.BASE
    exh = Exhaustivness.SMALLEST
    results = []
    for n in n_attackers:
        contexts = np.load(prefix+f"results/contexts/n_attacker={n}.npy")
        variables = get_SMK_V(n)[:-1]
        data = {
                "n_attacker": int(n),
                "beam_size": None,
                "exhaustiveness": exh.value,
                "model": model.value,
                "algo": "ILP",
                "results": []
            }
        for u in tqdm(contexts):
            u = [int(elt) for elt in u]
            s = SMK_model(u, {}, n)
            (C, W), t = time_fn(run_ilp_SMK, n, s, u, prefix=prefix)
            data["results"].append({
                    "rules": [serialize_symbols_values(C, W, s, variables)],
                    "causes": [serialize_symbols(C, variables)],
                    "context": u,
                    "time": t,
                })
        results.append(data)
    save_json(prefix+f"results/{model.value}-{exh.value}/ILP.json", results)
