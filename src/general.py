import numpy as np, json, os
from tqdm import tqdm

from benchmark_models import SMK_model, get_SMK_V

from collections.abc import Iterable
from enum import Enum

# === Hyper parameters ===
# == General ==
N = 20
n_attackers = (2,5,10)
n_attackers_smallest = [2,  4,  6,  8, 11, 13, 15, 17, 20]
full_attackers = list(set(n_attackers) | set(n_attackers_smallest))
beam_sizes = [ 1, 12, 25, 37, 50]
beam_sizes_smallest = [  1,  11,  22,  33,  44,  55,  66,  77,  88, 100]
max_steps = 8

# == Noisy global params ==
nl = 1.5
n_seeds = 10

# == LUCB params ==
max_iter_noisy = 20

eps=.65
lucb_params = {"a": .65, 
               "cause_eps": .1, 
               "non_cause_eps": .1, 
               "beam_eps": .1, 
               "max_iter": max_iter_noisy, 
               "verbose": 0, 
               "init_batch_size": 30,
               "batch_size": 10,
               "delta": .05
               }

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
            s_rule.append([dim, value])
        rules.append(s_rule)
    return rules

def serialize_interventions(E):
    out = []
    for e in E:
        s_e = []
        for dim, value in e:
            if isinstance(value, Iterable):
                value = [float(v) for v in value]
            else: value = float(value)
            s_e.append([dim, value])
        out.append(s_e)
    return out

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


# Contexts
def generate_base_contexts_SMK(n_attacker, N, seed=42):
    if seed is not None: np.random.seed(seed)
    exo_vars = ("fs", "fn", "ff", "fdb", "a", "ad")
    n_exo_vars = n_attacker * len(exo_vars)
    n_endo_vars = len(get_SMK_V(n_attacker))
    contexts = np.zeros((N,n_exo_vars), dtype=int)
    for n in tqdm(range(N)):
        while True:
            ids = np.arange(n_exo_vars)
            np.random.shuffle(ids)
            contexts[n][ids[:n_exo_vars//2]] = 1
            s = SMK_model(contexts[n], {}, n_attacker)
            if s[-1] and not any([(contexts[n] == contexts[i]).all() for i in range(n-1)]): break
    return contexts

def make_base_contexts(N, n_attackers, prefix=""):
    os.makedirs(prefix+"results/contexts",exist_ok=True)
    for n_attacker in n_attackers:
        if os.path.isfile(prefix+f"results/contexts/{n_attacker=}.npy"): continue
        contexts = generate_base_contexts_SMK(n_attacker, N, seed=42)
        np.save(prefix+f"results/contexts/{n_attacker=}.npy", contexts)
