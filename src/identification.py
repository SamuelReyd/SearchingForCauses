import numpy as np, json
from tqdm import tqdm

from itertools import product

from binary_models import *
from benchmark_models import SMK_model, get_SMK_dim_labels, get_mSMK_SCM, get_bbSMK_SCM, get_nSMK_SCM
from actualcauses import beam_search, iterative_identification

from ILP_why import run_ilp_SMK
from general import *

def run_one_SMK(contexts, n_attacker, beam_size, 
                exh: Exhaustivness, model: Models, struct: AlgoTypes, 
                max_steps=-1, verbose=0):
    variables = get_SMK_dim_labels(n_attacker)
    data = {
        "n_attacker": n_attacker,
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
        if model == Models.BASE:
                SCM = make_SCM(variables, u, SMK_model, n_attacker=n_attacker)
                dag, init_var_ids = build_DAG(n_attacker, SCM["variables"])
        elif model == Models.NON_BOOLEAN:
            SCM = get_mSMK_SCM(n_attacker, u)
            dag, init_var_ids = build_DAG_non_boolean(n_attacker, SCM["variables"])
        elif model == Models.NOISY:
            SCM = get_nSMK_SCM(n_attacker, u)
        elif model == Models.BLACK_BOX:
            SCM = get_bbSMK_SCM(n_attacker, u)
            dag = [[] for v in SCM["variables"]]
            init_var_ids = range(len(SCM["variables"]))
        # Adapt on algo type
        if struct == AlgoTypes.BASE:
            values, t = time_fn(
                beam_search, **SCM, 
                early_stop=early_stop, beam_size=beam_size, verbose=0
            )
        elif struct == AlgoTypes.STRUCTURED:
            values, t = time_fn(
                iterative_identification, **SCM, 
                dag=dag, init_var_ids=init_var_ids, 
                early_stop=early_stop, beam_size=beam_size, verbose=0
            )
        
        states = [value[-1] for value in values]
        causes = [tuple(value[3]) for value in values]
        s = SCM["instance"]
        data["results"].append({
            "rules": serialize_rules(values),
            "causes": causes,
            "context": u,
            "actual_state": s,
            "states": states,
            "time": t,
            "scores": [float(value[2]) for value in values]
        })
        # print(data)
    return data

def run_SMK(n_attackers, beam_sizes, 
                 exh: Exhaustivness, model: Models, struct: AlgoTypes, 
                 max_steps=-1, prefix=""):
    results = []
    all_contexts = {n_attacker: 
                np.load(prefix+f"results/contexts/{n_attacker=}.npy")
               for n_attacker in n_attackers}
    print("Indentify for:")
    print(f"  Exhaustiveness: {exh.value}")
    print(f"  Version of SMK: {model.value}")
    print(f"  Algorithm used: {struct.value}")
    print(f"{n_attackers=} and {beam_sizes=}")
    for n_attacker, beam_size in tqdm(list(product(n_attackers, beam_sizes))):
        contexts = all_contexts[n_attacker]
        data = run_one_SMK(contexts, n_attacker, beam_size, 
                           exh, model, struct,
                           max_steps, verbose=0)
        results.append(data)
    save_json(prefix+f"results/{model.value}-{exh.value}/{struct.value}.json", results)

def run_one_noisy_SMK(u, beam_size, algo, do_lucb, cause_eps, 
                      non_cause_eps, beam_eps, batch_size, nl, N, eps, 
                      seed, max_steps):
    early_stop = False
    n_attacker = len(u) // 6

    lucb_params = build_lucb_params(beam_size, do_lucb, cause_eps, 
                                    non_cause_eps, beam_eps, 
                                    batch_size, nl, N, eps)
    
    SCM = get_nSMK_SCM(n_attacker, u, do_lucb=do_lucb, 
                       N=N, nl=nl, lucb_params=lucb_params)
    dag, init_var_ids = build_DAG(n_attacker, SCM["variables"])
    np.random.seed(seed)
    if algo == AlgoTypes.BASE:
        values, t = time_fn(
            beam_search, 
            **SCM, 
            epsilon=eps,
            early_stop=early_stop, beam_size=beam_size, 
            verbose=0
        )
    else:
        values, t = time_fn(
            structured_identification, 
            **SCM, 
            epsilon=eps,
            dag=dag, init_var_ids=init_var_ids,
            early_stop=early_stop, beam_size=beam_size, 
            verbose=0
        )
    
    states = [value[-1] for value in values]
    causes = [tuple(value[3]) for value in values]
    s = SCM["instance"]
    return {
        "rules": serialize_rules(values),
        "causes": causes,
        "context": u,
        "actual_state": s,
        "states": states,
        "time": t,
        "seed": seed,
        "scores": [float(value[2]) for value in values],
        "lucb_infos": lucb_params["lucb_infos"] if lucb_params is not None else None
    }

def run_noisy_SMK(algo, n_attackers, beam_sizes, do_lucb,
                  cause_eps, non_cause_eps, beam_eps, batch_size, nl, N, eps,
                  seed=42, max_steps=-1, prefix=""):
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
    for n_attacker, beam_size in tqdm(list(product(n_attackers, beam_sizes))):
        contexts = all_contexts[n_attacker]
        
        variables = get_SMK_dim_labels(n_attacker)
        data = {
            "n_attacker": n_attacker,
            "beam_size": beam_size,
            "exhaustiveness": Exhaustivness.FULL.value,
            "model": Models.NOISY.value,
            "algo": algo.value,
            "do_lucb": do_lucb,
            "cause_eps": cause_eps,
            "non_cause_esp": non_cause_eps,
            "beam_eps": beam_eps,
            "batch_size": batch_size,
            "nl":nl,
            "N": N,
            "eps":eps,
            "seed":seed,
            "results": []
        }
        for u in contexts:
            u = u.tolist()
            data["results"].append(
                run_one_noisy_SMK(u, beam_size, algo, do_lucb, cause_eps, 
                                  non_cause_eps, beam_eps, batch_size, nl, N, eps, 
                                  seed, max_steps)
            )
        results.append(data)
    save_json(
        prefix+f"results/{Models.NOISY.value}-{Exhaustivness.FULL.value}/{algo.value}-{lucb_label}.json", 
        results
    )

def run_noisy_SCM_params(base_params, Ns, batch_sizes, algo,
                         u=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                         beam_size=50, N_exp=50, prefix=""
                        ):
    data = []
    for N in tqdm(Ns):
        for do_lucb in (True, False):
            if do_lucb: bs = batch_sizes
            else: bs = [None]
            for batch_size in bs:
                params = base_params | {
                    "beam_size": beam_size,
                    "N": N,
                    "batch_size": batch_size,
                    "do_lucb": do_lucb,
                }
                lucb_params = build_lucb_params(**params)
                datum = {
                    **params,
                    "lucb_params": lucb_params,
                    "u": u,
                    "n_attacker": len(u)//6,
                    "results": []
                }
                for seed in range(N_exp):
                    res = run_one_noisy_SMK(
                    u, **params, algo=algo,
                    seed=seed, max_steps=-1
                )
                    datum["results"].append(res)
                data.append(datum)
    save_json(prefix+f"results/noisy-params/{algo.value}.json", data)


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
    for n_attacker in n_attackers:
        contexts = np.load(prefix+f"results/contexts/{n_attacker=}.npy")
        variables = get_SMK_dim_labels(n_attacker)[:-1]
        data = {
                "n_attacker": int(n_attacker),
                "beam_size": None,
                "exhaustiveness": exh.value,
                "model": model.value,
                "algo": "ILP",
                "results": []
            }
        for u in tqdm(contexts):
            u = [int(elt) for elt in u]
            s = [0] * (len(variables)+1)
            SMK_model(s, u, {}, n_attacker)
            (C, W), t = time_fn(run_ilp_SMK,n_attacker, s, u, prefix=prefix)
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

def simulate_SMK_heuristic(rules, u, n_attacker, instance, heuristic):
    output = []
    variables = get_SMK_dim_labels(n_attacker)
    for rule in rules:
        cf = dict(rule)
        s = [0 for _ in variables]
        SMK_model(s, u, cf, n_attacker)
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

def get_simulation(u, n_attacker, instance, heuristic):
    return lambda rules: simulate_SMK_heuristic(rules, u, n_attacker, instance, heuristic)

def evaluate_heuristics(n_attacker, N, measure, prefix=""):
    contexts = np.load(prefix+f"results/contexts/{n_attacker=}.npy")
    variables = get_SMK_dim_labels(n_attacker)[:-1]
    domains = [(0,1)] * len(variables)
    dag, init_variables = build_DAG(n_attacker, variables)
    heuristics = get_heuristics(None)
    measures = {name:[] for name in heuristics}
    
    for context in tqdm(contexts[:N]):
        instance = [0] * (len(variables)+1)
        SMK_model(instance,context,{},n_attacker)
        simulation = get_simulation(context, n_attacker, instance, heuristics["heuristic_const"])
        ref_causes = structured_identification(instance, domains, simulation, variables, 
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