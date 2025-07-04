import numpy as np, timeit, time, matplotlib.pyplot as plt, json, os
from tqdm import tqdm
from matplotlib.lines import Line2D
from collections.abc import Iterable
from collections import defaultdict
from itertools import product
from enum import Enum

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from binary_models import *
from benchmark_models import rock_throwing_model, SMK_model, get_SMK_dim_labels, get_mSMK_SCM, get_bbSMK_SCM
from benchmark_models import get_noisy_suzy_SCM, get_nSMK_SCM
from actualcauses import beam_search, show_rules, iterative_identification
from actualcauses.base_algorithm import filter_minimality
from ILP_why import run_ilp_SMK

# General
class AlgoTypes(Enum):
    STRUCTURED = "structured"
    BASE = "base_algo"

class Exhaustivness(Enum):
    SMALLEST = "smallest"
    FULL = "full"
    EXACT = "exact"

class Models(Enum):
    BASE = "base"
    NON_BOOLEAN = "non-boolean"
    NOISY = "noisy"
    BLACK_BOX = "black-box"

def serialize_rules(rule_values):
    rules = []
    for rule_value in rule_values:
        rule = rule_value[0]
        s_rule = []
        for dim, value in rule:
            if isinstance(value, Iterable):
                value = [float(v) for v in value]
            else: value = float(value)
            s_rule.append([int(dim), value])
        rules.append(s_rule)
    return rules

def load_json(file):
    with open(file) as file:
        return json.load(file)

def save_json(path, data):
    folder = "/".join(path.split("/")[:-1])
    os.makedirs(folder,exist_ok=True)
    with open(path, "w") as file:
        file.write(json.dumps(data, indent=2))

def get_struct_label(struct):
    return "structured" if struct else "base_algo"

def build_lucb_params(beam_size, do_lucb, cause_eps, 
                      non_cause_esp, beam_eps, batch_size, nl, N, eps):
    if do_lucb:
        lucb_params = {
            "lucb_infos": [],
            "beam_size": beam_size,
            "max_iter": N,
            "a": eps,
            "cause_eps": cause_eps,
            "non_cause_esp": non_cause_esp,
            "beam_eps": beam_eps,
            "batch_size": batch_size,
            "verbose": 0
        }
    else: lucb_params = None
    return lucb_params


# Contexts
def generate_base_contexts_SMK(n_attacker, N, seed=42):
    if seed is not None: np.random.seed(seed)
    exo_vars = ("fs", "fn", "ff", "fdb", "a", "ad")
    n_exo_vars = n_attacker * len(exo_vars)
    n_endo_vars = len(get_SMK_dim_labels(n_attacker))
    contexts = np.zeros((N,n_exo_vars), dtype=int)
    s = np.ones(n_endo_vars)
    for n in tqdm(range(N)):
        while True:
            ids = np.arange(n_exo_vars)
            np.random.shuffle(ids)
            contexts[n][ids[:n_exo_vars//2]] = 1
            SMK_model(s, contexts[n], {}, n_attacker)
            if s[-1] and not any([(contexts[n] == contexts[i]).all() for i in range(n-1)]): break
    return contexts

def make_base_contexts(N, n_attackers, prefix=""):
    os.makedirs(prefix+"results/contexts",exist_ok=True)
    for n_attacker in n_attackers:
        if os.path.isfile(prefix+f"results/contexts/{n_attacker=}.npy"): continue
        contexts = generate_base_contexts_SMK(n_attacker, N, seed=42)
        np.save(prefix+f"results/contexts/{n_attacker=}.npy", contexts)


# Identification
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

# Evaluation
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

# Compare heuristics
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


# Plots
def plot_metric(metric, n_attackers, beam_sizes, 
                exh: Exhaustivness, model: Models,
                lucb_label=None,
                prefix="", ax=None, save=False):
    if ax is None:
        ax = plt.gca()
        show = True
    else: show = False
    if lucb_label is None: lab = ""
    else: lab = "-" + lucb_label
    att = {value: i for i, value in enumerate(n_attackers)}
    bs = {value: i for i, value in enumerate(beam_sizes)}
    for ls, struct in zip(("--", "-"), AlgoTypes):
        if os.path.isfile(prefix+f"results/{model.value}-{exh.value}/{struct.value}{lab}.json"):
            data = load_json(prefix+f"results/{model.value}-{exh.value}/{struct.value}{lab}.json")
        else:
            continue
        perfs = np.zeros((len(beam_sizes), len(n_attackers)))
        stds = np.zeros((len(beam_sizes), len(n_attackers)))
        for datum in data:
            if datum["beam_size"] == -1: continue
            perfs[bs[datum["beam_size"]], att[datum["n_attacker"]]] = datum[f"{metric}-avg"]
            stds[bs[datum["beam_size"]], att[datum["n_attacker"]]] = datum[f"{metric}-std"]
        for j, n_attacker in enumerate(n_attackers):
            ax.plot(np.array(beam_sizes) + j, perfs[:,j], 
                     color=f"C{j}", marker="x", ls=ls)
    
    ax.set_xlabel("Beam sizes")
    ax.set_ylabel(metric)

def plot_two_metrics(n_attackers, beam_sizes, metrics, title, prefix="../"):
    _, axes = plt.subplots(2, 5, figsize=(15,4.5), sharex=True,sharey="row")
    for measure, ax_line in zip(metrics,axes):
        line_gen = iter(ax_line)
        for model, ax in tqdm(list(zip((Models.BASE, Models.NON_BOOLEAN, Models.BLACK_BOX),line_gen))):
            plot_metric(measure, n_attackers, beam_sizes, 
                            Exhaustivness.FULL, model,
                            prefix="../", ax=ax)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.grid(axis="y")
        for lucb_label, ax in tqdm(list(zip(('lucb','naive'),line_gen))):
            plot_metric(measure, n_attackers, beam_sizes, 
                            Exhaustivness.FULL, Models.NOISY, lucb_label=lucb_label,
                            prefix="../", ax=ax)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.grid(axis="y")
    ax_gen = iter(axes[0])
    for ax, model in zip(axes[0,:3], (Models.BASE, Models.NON_BOOLEAN, Models.BLACK_BOX)):
        ax.set_title(model.value)
    for ax, lucb_label in zip(axes[0,3:5], ('lucb','naive')):
        ax.set_title(f"noisy - {lucb_label}")
    for ax, metric in zip(axes[:,0], metrics):
        ax.set_ylabel(metric)
    for ax in axes[1]: 
        if metrics[1] == "time":
            ax.set_yscale("log")
        ax.set_xticks(beam_sizes)
        ax.set_xlabel("Beam size")
        
    # Plot legend n_attacker
    for n_attacker in n_attackers:
        axes[1,0].plot([],[],label=f"n={n_attacker}")
    axes[1,0].legend(ncols=2)
    # Plot legend algo type
    for algo, ls in zip(AlgoTypes, ("--", "-")):
        axes[1,2].plot([],[],label=f"{algo.value}", ls=ls, c="grey")
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig(prefix+f"results/{title}.pdf")
    plt.show()
    


def plot_distributions(models, n_attackers, algo, exh, metric, prefix="../", base_size=2, plot_type="violin"):
    _, axes = plt.subplots(len(models),len(n_attackers), 
                           figsize=(base_size*len(n_attackers),base_size*len(models)), 
                           sharex=True, sharey='row', squeeze=False)
    
    for ax_line, (model, lucb_label) in tqdm(list(zip(axes, models))):
        for ax, n_attacker in zip(ax_line, n_attackers):
            plot_distribution(metric, n_attacker, algo, exh, 
                              model, lucb_label, ax, prefix, plot_type)
    for i, n_attacker in enumerate(n_attackers):
        axes[0,i].set_title(f"{n_attacker=}")
        axes[-1,i].set_xlabel("Beam sizes")
    for j, (model, lucb_label) in enumerate(models):
        lab = f" - {lucb_label}" if lucb_label else ""
        axes[j,0].set_ylabel(f"{model.value}{lab}\n\n{metric}")
    plt.tight_layout()
    plt.savefig(prefix+f"results/distributions-{algo.value}-{metric}.pdf")
    plt.show()

def plot_distribution(metric, n_attacker, algo, exh, model, lucb_label, ax=None, prefix="../", plot_type="violin"):
    if ax is None:
        ax = plt.gca()

    lab = f"-{lucb_label}" if lucb_label else ""
    data = load_json(prefix+f"results/{model.value}-{exh.value}/{algo.value}{lab}.json")
    values = {}
    for datum in data:
        if datum["beam_size"] == -1 or datum["n_attacker"] != n_attacker: continue
        values[datum["beam_size"]] = [res[metric] for res in datum["results"]]
    
    X_val = sorted(values.keys())
    Y = [values[x] for x in X_val]
    X = np.arange(len(X_val))
    if plot_type == "violin":
        ax.violinplot(Y, positions=X)
    elif plot_type == "box":
        ax.boxplot(Y, positions=X)
    ax.plot(X, np.mean(Y, axis=1), "x", c="black")
    ax.set_xticks(X, X_val)

def plot_noisy_params(Ns, prefix="../"):
    _, axes = plt.subplots(2,2, sharex=True)
    plot_noisy_params_metric(Ns, "F1", AlgoTypes.BASE, axes[0,0], prefix)
    plot_noisy_params_metric(Ns, "time", AlgoTypes.BASE, axes[1,0], prefix)
    plot_noisy_params_metric(Ns, "F1", AlgoTypes.STRUCTURED, axes[0,1], prefix)
    plot_noisy_params_metric(Ns, "time", AlgoTypes.STRUCTURED, axes[1,1], prefix)
    axes[0,0].set_title("Base algo")
    axes[0,1].set_title("Structured algo")
    for i in range(2):
        axes[1,i].set_xlabel("# sample per element")
    axes[0,1].legend()
    axes[0,0].set_ylabel("F1 score")
    axes[1,0].set_ylabel("time (s)")
    plt.tight_layout()
    plt.savefig(prefix+"results/params.pdf")
    plt.show()

def plot_noisy_params_metric(Ns, metric, algo, ax, prefix="../"):
    data = load_json(prefix+f"results/noisy-params/{algo.value}.json")
    perfs = defaultdict(lambda: np.zeros(len(Ns)))
    stds = defaultdict(lambda: np.zeros(len(Ns)))
    for datum in data:
        key = (datum['do_lucb'],datum['batch_size'])
        perfs[key][Ns.index(datum["N"])] = datum[f"{metric}-avg"]
        stds[key][Ns.index(datum["N"])] = datum[f"{metric}-std"]
    for key, value in perfs.items():
        label = f"batch size={key[1]}" if key[0] else "naive"
        ax.errorbar(Ns, value, yerr=stds[key], label=label, capsize=2, marker="x",elinewidth=1)
    ax.set_xticks(Ns)

def gather_data_smallest(prefix, algos, metrics, n_attackers, beam_sizes):
    all_perfs = {}
    all_stds = {}
    model = Models.BASE
    exh = Exhaustivness.SMALLEST
    for metric in tqdm(metrics):
        for algo_val in algos:
            data = load_json(prefix+f"results/{model.value}-{exh.value}/{algo_val}.json")
            perfs = np.zeros((len(beam_sizes), len(n_attackers)))
            stds = np.zeros((len(beam_sizes), len(n_attackers)))
            for datum in data:
                if datum["beam_size"] == -1: continue
                if datum["beam_size"] is None:
                    perfs[:, n_attackers.index(datum["n_attacker"])] = datum[f"{metric}-avg"]
                    stds[:, n_attackers.index(datum["n_attacker"])] = datum[f"{metric}-std"]
                else:
                    perfs[beam_sizes.index(datum["beam_size"]), n_attackers.index(datum["n_attacker"])] = datum[f"{metric}-avg"]
                    stds[beam_sizes.index(datum["beam_size"]), n_attackers.index(datum["n_attacker"])] = datum[f"{metric}-std"]
                all_perfs[f"{metric}-{algo_val}"] = perfs
                all_stds[f"{metric}-{algo_val}"] = stds
    return all_perfs, all_stds

def get_smallest_legend(ax):
    lines = []
    lines.append(Line2D([], [], label="Base algo", color="grey", ls="-", marker="x"))
    lines.append(Line2D([], [], label="Structured", color="grey", ls="--", marker="+"))
    lines.append(Line2D([], [], label="ILP", color="grey", ls="-.", marker="D"))
    return ax.legend(handles=lines, loc="lower left")

def plot_smallest_data(axes, metrics, Z, X, all_perfs, markers, lines, algos, xlabel, klabel):
    Z_ids = np.linspace(0,len(Z)-1,3,dtype=int)
    for ax, metric in zip(axes, metrics):
        for i, Z_i in enumerate(Z_ids):
            c_pad = 0 if xlabel == "beam sizes" else 3
            c = f"C{i+c_pad}"
            # Make the legends
            ax.plot([], label=f"{klabel}={Z[Z_i]}", color=c)
            
            for marker, ls, algo_val in zip(markers, lines, algos):
                # Do the plots
                perfs = all_perfs[f"{metric}-{algo_val}"]
                if perfs[Z_i].size != len(X): perfs = perfs.T
                if algo_val == "ILP": c = "grey"
                ax.plot(X, perfs[Z_i], color=c, marker=marker, ls=ls)
                       
            
        ax.set_xticks(X)
        
        if metric == 'time':
            ax.set_yscale("log")
    axes[0].legend(loc='upper left')
    axes[-1].set_xlabel(xlabel)

def plot_smallest(n_attackers, beam_sizes):
    exh = Exhaustivness.SMALLEST
    model = Models.BASE
    metrics = ("accuracy","time")
    prefix = "../"
    
    _, axes = plt.subplots(2,2,figsize=(7,4), sharey='row', sharex='col')
    legend_ax = axes[0,1]
    legend = get_smallest_legend(legend_ax)
    
    markers = ("x","+","D")
    lines = ("-", "--","-.")
    algos = (AlgoTypes.BASE.value,AlgoTypes.STRUCTURED.value,"ILP")

    all_perfs, all_stds = gather_data_smallest(prefix, algos, metrics, n_attackers, beam_sizes)

    plot_smallest_data(axes[:,0], metrics, n_attackers, beam_sizes, all_perfs, markers, lines, algos, "beam sizes","n")
    plot_smallest_data(axes[:,1], metrics, beam_sizes, n_attackers, all_perfs, markers, lines, algos, "# attackers", "b")

    legend_ax.add_artist(legend)
    for metric, ax_line in zip(metrics,axes):
        ax_line[0].set_ylabel(metric)
    plt.tight_layout()
    plt.savefig("../results/main-smallest.pdf")
    plt.show()

def plot_heuristics(prefix=""):
    measures = load_json(prefix+f"results/base-full/heuristics.json")
    X = list(get_heuristics(None).keys())
    X_pos = np.arange(len(X))
    Y = [measures[x] for x in X]
    Y_mean = [np.mean(measures[x]) for x in X]
    # Y_min = [np.std(measures[x]) for x in X]
    # Y_max = [np.std(measures[x]) for x in X]
    X = [x.replace("heuristic_", "") for x in X]
    plt.boxplot(Y, positions=X_pos)
    plt.plot(X_pos, Y_mean, "x")
    plt.ylabel("F1 scores")
    plt.xticks(X_pos, X, rotation=45)
    plt.savefig(prefix+"results/heuristics.pdf", bbox_inches='tight')
    plt.show()

def regression(x, y, degree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_pred)
    coefficients = model.named_steps['linearregression'].coef_
    return coefficients, r2, y_pred

def plot_regression(X, Y, Y_err, ax, i, z_label, degree):
    coefficients, r2, Y_pred = regression(X, Y, degree)
    if Y_err is None:
        ax.plot(X,Y, "x", c="black")
    else:
        ax.errorbar(X,Y, Y_err, c="black", ls="", marker="x")
    ax.plot(X,Y_pred, c="C0")
    ax.set_xticks(np.arange(X.size))
    ax.set_title(f"{z_label}\n$R^2={r2:.3f}$ - coef={coefficients[-1]:.2f}")

def plot_regressions(all_perfs, all_stds, algo_val, X, Z, z_lab, x_lab, transpose, do_ILP, 
                     n_col, n_row, degree, prefix="../", exh=Exhaustivness.SMALLEST):
    perf = all_perfs[f"time-{algo_val}"]
    stds = all_stds[f"time-{algo_val}"]
    acc = all_perfs[f"accuracy-{algo_val}"]
    if transpose:
        perf = perf.T
        stds = stds.T
        acc = acc.T
    _, axes = plt.subplots(n_row,n_col,figsize=(n_col*3,n_row*3), sharex=True)
    x = np.arange(len(X))

    ax_gen = iter(axes.flatten())
    for i, ax in zip(range(perf.shape[0]),ax_gen):
        y = perf[i]
        y_err = stds[i]
        plot_regression(x, y, y_err, ax, i, f"{z_lab}: {Z[i]}", degree)
        ax2 = ax.twinx()
        ax2.plot(x, acc[i], "--", c="grey")
        ax2.set_ylim(0,1.1)
        ax2.set_zorder(1)
        ax2.set_ylabel("Accuracy")
        ax.set_ylabel("Time (s)")
        
    for i in range(n_col):
        axes[1,i].set_xticks(x, X)
        axes[1,i].set_xlabel(x_lab)

    if do_ILP:
        plot_regression(x, all_perfs[f"time-ILP"][0], None, next(ax_gen), perf.shape[0], "ILP", degree)
    ax = next(ax_gen)
    ax.set_axis_off()
    ax.plot([], [], "x", color="black", label="Runtime")
    ax.plot([], [], "-", color="C0", label="Regression")
    ax.plot([], [], "--", color="grey", label="Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(prefix+f"results/smallest-time-vs-{x_lab}.pdf")
    plt.show()

def plot_regression_smallest(n_attackers, beam_sizes, metrics, prefix="../"):
    algos = (AlgoTypes.BASE.value,AlgoTypes.STRUCTURED.value,"ILP")
    
    all_perfs, all_stds = gather_data_smallest(prefix, algos, metrics, n_attackers, beam_sizes)

    plot_regressions(all_perfs, all_stds, "base_algo", 
                 n_attackers, beam_sizes, "beam size", "n attackers",
                 transpose=False, do_ILP=True, n_col=6, n_row=2, 
                 degree=2, prefix=prefix, exh=Exhaustivness.SMALLEST)

    plot_regressions(all_perfs, all_stds, "base_algo", 
                 beam_sizes, n_attackers, "n attackers", "beam size", 
                 transpose=True, do_ILP=False, n_col=5, n_row=2, 
                 degree=1, prefix=prefix, exh=Exhaustivness.SMALLEST)

# Define experiments and parameters
N = 50
n_attackers = (2,5,10)
n_attackers_smallest = [2,  4,  6,  8, 11, 13, 15, 17, 20]
full_attackers = list(set(n_attackers) | set(n_attackers_smallest))
beam_sizes = [ 1, 12, 25, 37, 50]
beam_sizes_smallest = [  1,  11,  22,  33,  44,  55,  66,  77,  88, 100]

exps = (
        (Exhaustivness.FULL, Models.BASE, AlgoTypes.BASE),
        (Exhaustivness.FULL, Models.BASE, AlgoTypes.STRUCTURED),
        (Exhaustivness.FULL, Models.NON_BOOLEAN, AlgoTypes.BASE),
        (Exhaustivness.FULL, Models.NON_BOOLEAN, AlgoTypes.STRUCTURED),
        (Exhaustivness.FULL, Models.BLACK_BOX, AlgoTypes.BASE),
    )

base_params = {
        "do_lucb": True,
        "cause_eps": .01,
        "non_cause_eps": .01,
        "beam_eps": .1,
        "batch_size": 10,
        "nl": .01,
        "N": 20,
        "eps": .3
    }
noisy_beam_size = 25
N_seeds = 50
Ns = np.linspace(1,50,5, dtype=int).tolist()
batch_sizes = [2, 5, 10]
    
# Main
if __name__ == "__main__":
    # Contexts
    make_base_contexts(N, full_attackers)

    # ILP
    if not os.path.isfile(f"results/base-smallest/ILP.json"):
        print("Run ILP")
        run_ILP_SMK(n_attackers_smallest, prefix="") 

    # Heuristics
    if not os.path.isfile(f"results/base-full/heuristics.json"):
        evaluate_heuristics(n_attacker=5, N=50, measure="F1", prefix="")
    
    # Exact results - base model
    if not os.path.isfile(f"results/{Models.BASE.value}-{Exhaustivness.EXACT.value}/{AlgoTypes.STRUCTURED.value}.json"):
        run_SMK(n_attackers, [-1], Exhaustivness.EXACT, Models.BASE, AlgoTypes.STRUCTURED)
    
    # Main experiments - detrministic
    for exh, model, algo in exps:
        if os.path.isfile(f"results/{model.value}-{exh.value}/{algo.value}.json"): 
            continue
        run_SMK(n_attackers, beam_sizes, exh, model, algo)
        print("Evaluation...")
        evaluate_SMK(model, exh)
        print()
        
    # Main experiments - noisy
    exh = Exhaustivness.FULL
    model = Models.NOISY
    for algo in AlgoTypes:
        for do_lucb in (True, False):
            lucb_label = "lucb" if do_lucb else "naive"
            if os.path.isfile(f"results/{model.value}-{exh.value}/{algo.value}-{lucb_label}.json"):
                continue
            params = base_params | {"do_lucb": do_lucb}
            run_noisy_SMK(algo, n_attackers, beam_sizes,**params)
    evaluate_noisy_SMK(prefix="")
    
    # Smallest identification
    for algo in AlgoTypes:
        if os.path.isfile(f"results/base-smallest/{algo.value}.json"):
            continue
        run_SMK(n_attackers_smallest, beam_sizes_smallest, Exhaustivness.SMALLEST, Models.BASE, algo)
        print()
    evaluate_SMK(Models.BASE, Exhaustivness.SMALLEST, prefix="")
    
    # Noisy experiments
    exh = Exhaustivness.FULL
    model = Models.NOISY
    for algo in AlgoTypes:
        if os.path.isfile(f"results/noisy-params/{algo.value}.json"):
            continue
        print(f"Run noisy params with {algo=}")
        run_noisy_SCM_params(base_params, Ns, batch_sizes, algo,
                             beam_size=noisy_beam_size, N_exp=N_seeds, prefix="")
    evaluate_params_SMK(algo, prefix="")
