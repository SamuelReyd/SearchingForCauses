import numpy as np, json
from tqdm import tqdm

from itertools import product

from binary_models import *
from benchmark_models import SMK_model, get_SMK_SCM, get_SMK_V, get_mSMK_SCM, get_bbSMK_SCM, get_avg_nSMK_SCM, get_lucb_nSMK_SCM
# from actualcauses import beam_search, iterative_identification
from actualcauses_local.base_algorithm import beam_search
from actualcauses_local.iterative_subinstance_algorithm import iterative_identification

from ILP_why import run_ilp_SMK
from general import *

# ==== Base run ===
def run_one_SMK(contexts, n, beam_size, 
                exh: Exhaustivness, model: Models, struct: AlgoTypes, 
                max_steps, verbose=0):

    data = {
        "n_attacker": n,
        "max_steps": max_steps,
        "beam_size": beam_size,
        "exhaustiveness": exh.value,
        "model": model.value,
        "algo": struct.value,
        "results": []
    }
    for u in tqdm(contexts, disable=not verbose):
        # Adapt on exhaustivness
        early_stop = (exh == Exhaustivness.SMALLEST)
        u = u.tolist()
        # Adapt on the version of the SCM  
        if model == Models.BASE: scm = get_SMK_SCM(n, u)
        elif model == Models.NON_BOOLEAN: scm = get_mSMK_SCM(n, u)
        elif model == Models.BLACK_BOX: scm = get_bbSMK_SCM(n, u)
        else: raise Exception(f"Model {model} not handled")
        use_ISI = struct==AlgoTypes.STRUCTURED
        
        scm.find_causes(ISI=use_ISI, beam_size=beam_size, max_steps=max_steps)
        
        data["results"].append({
            "rules": serialize_interventions(scm.interventions),
            "causes": scm.causes_hashable,
            "context": u,
            "time": scm.identification_time,
        })
    return data

def run_SMK(n_attackers, beam_sizes, 
            exh: Exhaustivness, model: Models, struct: AlgoTypes,
            max_steps, prefix=""):
    results = []
    all_contexts = {n_attacker: 
                np.load(prefix+f"results/contexts/{n_attacker=}.npy")
               for n_attacker in n_attackers}
    print("Indentify for:")
    print(f"  Exhaustiveness: {exh.value}")
    print(f"  Version of SMK: {model.value}")
    print(f"  Algorithm used: {struct.value}")
    print(f"{n_attackers=} and {beam_sizes=}")
    for n, beam_size in tqdm(list(product(n_attackers, beam_sizes))):
        contexts = all_contexts[n]
        data = run_one_SMK(contexts, n, beam_size, exh, model, struct, max_steps, verbose=0)
        results.append(data)
    save_json(prefix+f"results/{model.value}-{exh.value}/{struct.value}.json", results)

# === Noisy run ===
def run_one_noisy_SMK(n, u, beam_size, algo, max_steps, lucb_params, seed, nl, do_lucb):
    eps = lucb_params["a"]
    lucb_params = lucb_params | {"beam_size": beam_size}
    lucb_params["lucb_info"] = []
    
    if do_lucb: scm = get_lucb_nSMK_SCM(n, u, nl, lucb_params)
    else: scm = get_avg_nSMK_SCM(n, u, lucb_params["max_iter"], nl)
    np.random.seed(seed)

    do_ISI = (algo != AlgoTypes.BASE)
    scm.find_causes(ISI=do_ISI, epsilon=eps,beam_size=beam_size,max_steps=max_steps)
    
    return {
        "rules": serialize_interventions(scm.interventions),
        "causes": scm.causes_hashable,
        "context": u,
        "time": scm.identification_time,
        "seed": seed,
        "lucb_info": lucb_params["lucb_info"] if lucb_params is not None else None
    }

def run_noisy_SMK(algo, n_attackers, beam_sizes, max_steps, lucb_params, n_seeds, nl, do_lucb, prefix=""):
    results = []
    all_contexts = {n_attacker: 
                np.load(prefix+f"results/contexts/{n_attacker=}.npy")
               for n_attacker in n_attackers}
    lucb_label = "lucb" if do_lucb else "naive"
    print("Indentify for:")
    print(f"  Exhaustiveness: {Exhaustivness.FULL.value}")
    print(f"  Version of SMK: {Models.NOISY.value}")
    print(f"  Algorithm used: {algo.value} - {lucb_label}")
    print(f"{n_attackers=} and {beam_sizes=}")
    for n, beam_size in tqdm(list(product(n_attackers, beam_sizes))):
        contexts = all_contexts[n]
        data = {
            "n_attacker": n,
            "beam_size": beam_size,
            "max_steps": max_steps,
            "exhaustiveness": Exhaustivness.FULL.value,
            "model": Models.NOISY.value,
            "algo": algo.value,
            "nl": nl, 
            "lucb_params": lucb_params.copy(),
            "results": []
        }

        for u in contexts:
            u = u.tolist()
            for seed in range(n_seeds):
                data["results"].append(
                    run_one_noisy_SMK(n, u, beam_size, algo, max_steps, lucb_params, seed, nl, do_lucb)
                )
        results.append(data)
    save_json(
        prefix+f"results/{Models.NOISY.value}-{Exhaustivness.FULL.value}/{algo.value}-{lucb_label}.json", 
        results
    )

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
        contexts = np.load(prefix+f"results/contexts/{n_attacker=}.npy")
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
                    "rules": serialize_symbols_values(C, W, s, variables),
                    "causes": serialize_symbols(C, variables),
                    "context": u,
                    "actual_state": s,
                    "states": None,
                    "time": t,
                    "scores": None
                })
        results.append(data)
    save_json(prefix+f"results/{model.value}-{exh.value}/ILP.json", results)

# === Heuristics ===
def simulate_SMK_heuristic(rules, u, n, instance, heuristic):
    output = []
    for rule in rules:
        cf = dict(rule)
        s = SMK_model(u, cf, n_attacker)
        output.append((s, s[-1], heuristic(s)))
    return output

def get_heuristics(instance):
    return  {
        "heuristic_sum_pos": lambda s: sum(s) - 1,
        "heuristic_sum_dif": lambda s: sum([s_cf != s_act for s_cf, s_act in zip(s, instance)]),
        "heuristic_sum_neg": lambda s: len(s) - sum(s) + 1,
        "heuristic_okham": lambda s: len(s) - sum([s_cf != s_act for s_cf, s_act in zip(s, instance)]),
        "heuristic_rand": lambda s: int(np.random.randint(len(s))),
        "heuristic_const": lambda s: len(s)//2,
    }

def get_simulation(u, n, instance, heuristic):
    return lambda rules: simulate_SMK_heuristic(rules, u, n, instance, heuristic)

def evaluate_heuristics(n_attacker, N, measure, prefix=""):
    contexts = np.load(prefix+f"results/contexts/{n_attacker=}.npy")
    variables = get_SMK_V(n_attacker)[:-1]
    domains = [(0,1)] * len(variables)
    dag, init_variables = build_DAG(n_attacker, variables)
    heuristics = get_heuristics(None)
    measures = {name:[] for name in heuristics}
    
    for context in tqdm(contexts[:N]):
        instance = [0] * (len(variables)+1)
        SMK_model(instance,context,{},n_attacker)
        simulation = get_simulation(context, n_attacker, instance, heuristics["heuristic_const"])
        ref_causes = iterative_identification(instance, domains, simulation, variables, 
                                           dag=dag, init_var_ids=init_variables,
                                           max_steps=-1, beam_size=-1, early_stop=False, verbose=0)
        ref_causes = [tuple(c[3]) for c in ref_causes]
    
        heuristics = get_heuristics(instance)
        
        simulations = {name: get_simulation(context, n_attacker, instance, heuristic) for name, heuristic in heuristics.items()}
        for name, simulation in simulations.items():
            causes = beam_search(instance, domains, simulation, variables, max_steps=-1, beam_size=10, early_stop=False, verbose=0)
            causes = [tuple(c[3]) for c in causes]
            scores = evaluate_full(causes, ref_causes)
            measures[name].append(scores[measure])
    save_json(prefix+f"results/base-full/heuristics.json", measures)
