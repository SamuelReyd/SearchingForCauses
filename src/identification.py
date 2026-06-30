import numpy as np, json
from tqdm import tqdm

from itertools import product
from collections import defaultdict

from benchmark_models import SMKModel, get_SMK_SCM, get_SMK_V, get_mSMK_SCM, get_bbSMK_SCM, get_avg_nSMK_SCM, get_lucb_nSMK_SCM

from ILP_why import ilp_SMK
from general import *


heuristics_refs = {
    "Sum-pos": lambda s, v: np.sum(s[:, :-1], axis=1),
    "Sum-eq": lambda s, v: np.sum(s[:, :-1] == v[:-1], axis=1),
    "Sum-neg": lambda s, v: np.sum(1 - s[:, :-1], axis=1) - 1,
    "Occam": lambda s, v: np.sum(s[:, :-1] != v[:-1], axis=1),
    "Rand": lambda s, v: np.random.randint(s.shape[1], size=s.shape[0]),
    "Const": lambda s, v: np.full(s.shape[0], 42),
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
    # print(f"  Max n of steps: {max_steps}")
    print(f"{n_attackers=}, {beam_sizes=}, {heuristics=}")
    for n, bs, heuristic in tqdm(list(product(n_attackers, beam_sizes, heuristics))):
        contexts = np.load(folder+f"contexts/n_attacker={n}.npy")
        data = run_one_SMK(contexts, exh, model, algo, bs, n, heuristic, lucb_label, 
                           max_steps, lucb_params, nl, n_seeds, verbose=0)
        results.append(data)
    save_json(folder+file_name, results)
    
def run_one_SMK(contexts, exh, model, algo, bs, n, heuristic, lucb_label, 
                max_steps, lucb_params, nl, n_seeds, verbose=0):
    
    # max_steps = -1 if exh == Exhaustivness.EXACT else 2 * n + 1
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
        if heuristic is not None: scm = get_SMK_SCM(n, u, heuristic=heuristics_refs[heuristic])
        elif model == Models.BASE: scm = get_SMK_SCM(n, u)
        elif model == Models.NON_BOOLEAN: scm = get_mSMK_SCM(n, u)
        elif model == Models.BLACK_BOX: scm = get_bbSMK_SCM(n, u)
        elif model == Models.NOISY: 
            params = lucb_params | {"beam_size": bs}
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
            "n_calls": scm.n_calls,
            "context": u,
            "time": scm.identification_time,
            "seed": seed
        }
        data["results"].append(res)
    return data


# === ILP ===
def serialize_rule_ILP(C, W, instance, variables):
    var_map = {val: i for i, val in enumerate(variables)}
    rules = [(v.name,1-int(instance[var_map[v.name]])) for v in C]
    rules += [(v.name,int(instance[var_map[v.name]])) for v in W]
    return rules

def run_ILP_SMK(n_attackers, folder=""):
    model = Models.BASE
    exh = Exhaustivness.SMALLEST
    results = []
    prefix = "../" * folder.startswith("../")
    for n in n_attackers:
        contexts = np.load(folder+f"contexts/n_attacker={n}.npy")
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
            s = SMKModel(n)(u, {})
            (C, W), t = time_fn(ilp_SMK, n, s, u, prefix=prefix)
            data["results"].append({
                    "rules": [serialize_rule_ILP(C, W, s, variables)],
                    "causes": [[v.name for v in C]],
                    "context": u,
                    "time": t,
                })
        results.append(data)
    save_json(folder+f"{model.value}-{exh.value}/ILP.json", results)
