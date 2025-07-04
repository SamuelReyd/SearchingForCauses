import math, numpy as np
from itertools import combinations
from sympy import symbols, Equivalent, to_cnf, Or, And, Not
from actualcauses import lucb
from tqdm import tqdm

"""Rock Throwing"""
def rock_throwing_model(s, V_exo, cf):
    ST, BT, SH, BH, BS = range(len(s))
    s[ST] = cf.get(ST, V_exo[0])
    s[BT] = cf.get(BT, V_exo[1])
    s[SH] = cf.get(SH, s[ST])
    s[BH] = cf.get(BH, s[BT] and not s[SH])
    s[BS] = cf.get(BS, s[BH] or s[SH])

rock_throwing_variables = ["ST","BT","SH","BH","BS"]
rock_thowing_exo = (1,1)

"""Forest Fire"""
def forest_fire_model(s, V_exo, cf, disjunctive):
    L, MD, FF = range(len(s))
    s[L] = cf.get(L, V_exo[0])
    s[MD] = cf.get(MD, V_exo[1])
    s[FF] = cf.get(FF, 
            s[L] or s[MD] if disjunctive 
            else s[L] and s[MD])

forest_fire_variables=["L","MD","FF"], 
forest_fire_exo=(1,1)

"""Rock Throwing Extended"""
def ext_rock_throwing_model(s, V_exo, cf):
    ST, BT, W, SH, BH, BS = range(len(s))
    s[ST] = cf.get(ST, V_exo[0])
    s[BT] = cf.get(BT, V_exo[1])
    s[W] = cf.get(W, V_exo[2])
    s[SH] = cf.get(SH, s[ST] and not s[W])
    s[BH] = cf.get(BH, s[BT] and not s[SH] and not s[W])
    s[BS] = cf.get(BS, s[BH] or s[SH])

ext_rock_throwing_variables=["ST","BT","W","SH","BH","BS"] 
ext_rock_throwing_exo=(1,1,0)

"""Prisoners"""
def prisoners_model(s, V_exo, cf):
    A, B, C, D = range(len(s))
    s[A] = cf.get(A, V_exo[0])
    s[B] = cf.get(B, V_exo[1])
    s[C] = cf.get(C, V_exo[2])
    s[D] = cf.get(D, (s[A] and s[B]) or s[C])

prisoners_variables=["A","B","C", "D"]
prisoners_exo=(1,1,1)

"""Assassin"""
def assassin_model(s, V_exo, cf, variant):
    A, B, VS = range(len(s))
    s[B] = cf.get(B, V_exo[1])
    s[A] = cf.get(A, V_exo[0] if variant == 1 
                     else V_exo[0] and s[B])
    s[VS] = cf.get(VS, not s[A] or s[B])

assassin_variables=["A","B","VS"]
assassin_exo=(1,1)

"""Railroad"""
def railroad_model(s, V_exo, cf):
    LB, F, RB, A = range(len(s))
    s[LB] = cf.get(LB, V_exo[0])
    s[F] = cf.get(F, V_exo[1])
    s[RB] = cf.get(RB, V_exo[2])
    s[A] = cf.get(A, 
                  not (
                      (s[F] and s[RB]) or (not s[F] and s[LB])
                  ) 
                 )

railroad_variables=["LB","F", "RB","A"]
railroad_exo=(0,1,0)

"""Abstract Model 1"""
def first_abstract_model(s, V_exo, cf):
    A, B, C = range(len(s))
    s[A] = cf.get(A, V_exo[0])
    s[B] = cf.get(B, V_exo[1])
    s[C] = cf.get(C, (s[A] and s[B]) or (not s[A] and not s[B]))

first_abstract_variables=["A","B","C"]
first_abstract_exo=(1,1)

"""Abstract Model 2"""
def second_abstract_model(s, V_exo, cf):
    A, B, C, D, E, G, H, I = range(len(s))
    s[A] = cf.get(A, V_exo[0])
    s[B] = cf.get(B, V_exo[1] and not s[A])
    s[C] = cf.get(C, s[A] or s[B])
    s[D] = cf.get(D, s[A])
    s[E] = cf.get(E, not s[A])
    s[G] = cf.get(G, not s[C])
    s[H] = cf.get(H, not s[C] and not s[G])
    s[I] = cf.get(I, any([s[C], s[D], s[E], s[G], s[H]]))

second_abstract_variables=["A","B","C","D","E","G","H","I"]
second_abstract_exo=(1,1)

"""LSP Model"""
def lsp_model(s, V_exo, cf):
    dim_labels = [f"X{i}" for i in range(1,42)]
    dim2id = dict(zip(dim_labels, range(len(dim_labels))))
    for i in range(1,26):
        dim = dim2id[f"X{i}"]
        s[dim] = cf.get(dim, V_exo[dim])
    equations = [
        (27, "or", [3,4]),
        (28, "or", [5,6]),
        (29, "or", [7,8]),
        (30, "or", [9,10]),
        (31, "or", [12,13,14,15,16]),
        (32, "and", [18,19]),
        (33, "and", [20,21]),
        (34, "and", [22,23]),
        (35, "and", [24,25]),
        (36, "or", [27,28,29,30]),
        (37, "and", [31,17]),
        (38, "and", [1,2]),
        (39, "and", [36,11]),
        (40, "or", [37,32,33,34,35]),
        (41, "or", [38,39,40,26]),
    ]
    for target, connector, sources in equations:
        if connector == "or":
            value = any([s[dim2id[f"X{i}"]] for i in sources])
        else:
            value = all([s[dim2id[f"X{i}"]] for i in sources])
        dim = dim2id[f"X{target}"]
        s[dim] = cf.get(dim, value)

lsp_variables=[f"X{i}" for i in range(1,42)]

"""Binary trees"""
def get_n_leaves(depth):
    n_nodes = 2**depth-1
    return math.ceil((n_nodes + 1) / 2)
    
def binary_tree_model(s, V_exo, cf, depth):
    n_nodes = 2**depth-1
    n_leaves = math.ceil((n_nodes + 1) / 2)
    for i in range(n_leaves, n_nodes):
        s[i] = cf.get(i, V_exo[n_nodes-i])
    for i in range(n_nodes-n_leaves-1, -1, -1):
        s[i] = cf.get(i, s[2*i+1] or s[2*i+2])

"""Steal Master Key"""
def SMK_model(s, V_exo, cf, n_attacker):
    dim_labels = get_SMK_dim_labels(n_attacker)
    dim2id = dict(zip(dim_labels, range(len(dim_labels))))
    # Set variables from exo source
    for i in range(1,n_attacker+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            s[dim_id] = cf.get(dim_id, V_exo[dim_id])
    # Set depth-1
    for i in range(1,n_attacker+1):
        # Set GP
        s[dim2id[f"GP-U{i}"]] = cf.get(
            dim2id[f"GP-U{i}"], 
            s[dim2id[f"FS-U{i}"]] or s[dim2id[f"FN-U{i}"]] 
        )
        # Set GK
        s[dim2id[f"GK-U{i}"]] = cf.get(
            dim2id[f"GK-U{i}"], 
            s[dim2id[f"FF-U{i}"]] or s[dim2id[f"FDB-U{i}"]]
        )
        # Set KMS
        s[dim2id[f"KMS-U{i}"]] = cf.get(
            dim2id[f"KMS-U{i}"], 
            s[dim2id[f"A-U{i}"]] and s[dim2id[f"AD-U{i}"]]
        )
    # Set DK
    for i in range(1,n_attacker+1):
        s[dim2id[f"DK-U{i}"]] = cf.get(
            dim2id[f"DK-U{i}"], 
            s[dim2id[f"GP-U{i}"]] and \
            s[dim2id[f"GK-U{i}"]] and \
            not any([s[dim2id[f"DK-U{j}"]] for j in range(1, i)])
        )
    # Set SD
    for i in range(1, n_attacker+1):
        s[dim2id[f"SD-U{i}"]] = cf.get(
            dim2id[f"SD-U{i}"], 
            s[dim2id[f"KMS-U{i}"]] and \
            not any([s[dim2id[f"SD-U{j}"]] for j in range(1,i)])
        )
    # Set global DK
    s[dim2id["DK"]] = cf.get(
        dim2id["DK"], 
        any([s[dim2id[f"DK-U{i}"]] for i in range(1,n_attacker+1)])
    )
    # Set global SD
    s[dim2id["SD"]] = cf.get(
        dim2id["SD"], 
        any([s[dim2id[f"SD-U{i}"]] for i in range(1,n_attacker+1)])
    )
    # Set SMK
    s[dim2id["SMK"]] = cf.get(
        dim2id["SMK"], 
        s[dim2id["DK"]] or s[dim2id["SD"]]
    )

def get_SMK_dim_labels(n_attacker):
    attacker_vars = ["FS", "FN", "FF", "FDB", "A", "AD", 
                     "GP", "GK", "KMS", "DK", "SD"]
    return [
        f"{dim}-U{i}" for dim in attacker_vars for i in range(1,n_attacker+1)
    ] + ["DK", "SD", "SMK"]

def get_sympy_SMK(n_attacker):
    leaf_vars = ["FS", "FN", "FF", "FDB", "A", "AD"]
    node_vars = ["GP", "GK", "KMS", "DK", "SD"]
    
    exo_var_labels = [
        f"{dim.lower()}-u{i}" \
    for dim in leaf_vars \
    for i in range(1,n_attacker+1)
    ]
    
    var_labels = [
            f"{dim}-U{i}" \
        for dim in leaf_vars + node_vars \
        for i in range(1,n_attacker+1)
        ]
    
    var_labels += ["DK", "SD", "SMK"]
    var_labels += exo_var_labels
    
    label2id = {label: i for i,label in enumerate(var_labels)}
    
    variables = symbols(var_labels)
    
    target = variables[label2id["SMK"]]
    
    equations = {}
    # Set leaves
    for exo_var_label in exo_var_labels:
        exo_var_id = label2id[exo_var_label]
        endo_var_id = label2id[exo_var_label.upper()]
        equations[variables[endo_var_id]] = variables[exo_var_id]
        
    # Set depth 1
    for i in range(1,n_attacker+1):
        GP, FS, FN, GK, FF, FDB, KMS, A, AD = [
            variables[label2id[f"{var_name}-U{i}"]] for \
            var_name in "GP, FS, FN, GK, FF, FDB, KMS, A, AD".split(", ")
        ]
        
        equations[GK] = FF | FDB
        equations[KMS] = A & AD
        equations[GP] = FS | FN
        
    # Set DK
    for i in range(1,n_attacker+1):
        equations[variables[label2id[f"DK-U{i}"]]] = \
        variables[label2id[f"GP-U{i}"]] & variables[label2id[f"GK-U{i}"]] &\
        And(*[~variables[label2id[f"DK-U{j}"]] for j in range(1, i)])
    
    # Set SD
    for i in range(1,n_attacker+1):
        equations[variables[label2id[f"SD-U{i}"]]] = \
        variables[label2id[f"KMS-U{i}"]] &\
        And(*[~variables[label2id[f"SD-U{j}"]] for j in range(1, i)])
    
    # Set global DK
    equations[variables[label2id["DK"]]] = \
    Or(*[variables[label2id[f"DK-U{i}"]] for i in range(1, n_attacker+1)])
    
    
    # Set global SD
    equations[variables[label2id["SD"]]] = \
    Or(*[variables[label2id[f"SD-U{i}"]] for i in range(1, n_attacker+1)])
    
    # Set SMK
    equations[variables[label2id["SMK"]]] = \
    variables[label2id["SD"]] | variables[label2id["DK"]]
    return variables, equations, variables[label2id["SMK"]]

"""Non-boolean SMK"""
mSMK_variables = ["FS", "FN", "FF", "FDB", "A", "AD", "GP", "GK", "KMS", "DK", "SD", "SMK"]

def generate_subsets(n):
    subsets = []
    for length in range(n + 1):
        for subset in combinations(range(n), length):
            subsets.append(tuple(subset))
    return subsets

def mSMK_model(s, u, cf, n_attacker):
    dim2id = dict(zip(mSMK_variables, range(len(mSMK_variables))))
            
    # Set from exo
    labels = ("FS", "FN", "FF", "FDB", "A", "AD")
    for i, label in enumerate(labels):
        dim_id = dim2id[label]
        s[dim_id] = cf.get(dim_id, tuple([j for j in range(n_attacker) if u[i * n_attacker + j]]))
    
    # Set KMS
    s[dim2id["KMS"]] = cf.get(dim2id["KMS"], tuple(set(s[dim2id["A"]]) & set(s[dim2id["AD"]])))
    
    # Set SD
    s[dim2id["SD"]] = cf.get(dim2id["SD"], min(s[dim2id["KMS"]]) if s[dim2id["KMS"]] else -1)

    # Set GK 
    s[dim2id["GK"]] = cf.get(dim2id["GK"], tuple(set(s[dim2id["FF"]]) | set(s[dim2id["FDB"]])))
    
    # Set GP
    s[dim2id["GP"]] = cf.get(dim2id["GP"], tuple(set(s[dim2id["FS"]]) | set(s[dim2id["FN"]])))

    # Set DK
    children = tuple(set(s[dim2id["GP"]]) & set(s[dim2id["GK"]]))
    s[dim2id["DK"]] = cf.get(dim2id["DK"], min(children) if children else -1)
    
    # Set SMK
    s[dim2id["SMK"]] = cf.get(dim2id["SMK"], s[dim2id["DK"]] > -1 or s[dim2id["SD"]] > -1)

def get_mSMK_domains(n_attacker):
    return [
        generate_subsets(n_attacker) for var in ("FS", "FN", "FF", "FDB", "A", "AD", "GP", "GK", "KMS") 
    ] + [
        [-1] + list(range(n_attacker)),
        [-1] + list(range(n_attacker))
    ]

def mSMK_heuristic(s):
    dim2id = dict(zip(mSMK_variables, range(len(mSMK_variables))))
    return (
        sum([len(s[dim2id[dim]]) for dim in ("FS", "FN", "FF", "FDB", "A", "AD", "GP", "GK", "KMS")]) + 
        s[dim2id["DK"]] + s[dim2id["SD"]]
    )

def mSMK_simulation(rules, u, n_attacker, mSMK_variables, verbose=0):
    output = []
    for rule in tqdm(rules, disable=not verbose):
        s = np.zeros(len(mSMK_variables), dtype=object).tolist()
        mSMK_model(s, u, dict(rule), n_attacker)
        output.append((s, s[-1], mSMK_heuristic(s)))
    return output

def get_mSMK_SCM(n_attacker, u, verbose=0):
    domains = get_mSMK_domains(n_attacker)
    s = np.zeros(len(mSMK_variables), dtype=object).tolist()
    mSMK_model(s, u, {}, n_attacker)
    sim = lambda rules: mSMK_simulation(rules, u, n_attacker, mSMK_variables, verbose)
    return {
        "variables": mSMK_variables[:-1],
        "domains": domains, 
        "instance": s,
        "simulation": sim
    }

"""Black Box SMK"""
    
def get_bbSMK_variables(n_attackers):
    return [f"{label}-U{i}" for label in ("FS", "FN", "FF", "FDB", "A", "AD") for i in range(1,n_attackers+1)] + ["SMK"]

def bbSMK_model(s, u, cf, n_attacker):
    variables = get_bbSMK_variables(n_attacker)
    dim2id = dict(zip(variables, range(len(variables))))
            
    # Set from exo
    for i in range(1,n_attacker+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            s[dim_id] = cf.get(dim_id, u[dim_id])
    GP = [s[dim2id[f"FS-U{i}"]] or s[dim2id[f"FN-U{i}"]] for i in range(1, n_attacker+1)]
    GK = [s[dim2id[f"FF-U{i}"]] or s[dim2id[f"FDB-U{i}"]] for i in range(1, n_attacker+1)]
    KMS = [s[dim2id[f"A-U{i}"]] and s[dim2id[f"AD-U{i}"]] for i in range(1, n_attacker+1)]
    DK = any([gp and gk for gp, gk in zip(GP,GK)])
    SD = any(KMS)
        
    # print(f"{GP=}, {GK=}, {KMS=}, {DK=}, {SD=}")
    # Set SMK
    s[dim2id["SMK"]] = cf.get(
        dim2id["SMK"], 
        int(DK or SD)
    )

def bbSMK_simulation(rules, u, n_attacker, variables, verbose=0):
    output = []
    for rule in tqdm(rules, disable=not verbose):
        s = np.zeros(len(variables), dtype=object).tolist()
        bbSMK_model(s, u, dict(rule), n_attacker)
        output.append((s, float(s[-1]), sum(s)-1))
    return output

def get_bbSMK_SCM(n_attacker, u, verbose=0):
    variables = get_bbSMK_variables(n_attacker)
    domains =  [(0,1)] * 6 * n_attacker
    s = np.zeros(len(variables), dtype=object).tolist()
    bbSMK_model(s, u, {}, n_attacker)
    sim = lambda rules: bbSMK_simulation(rules, u, n_attacker, variables, verbose)
    return {
        "variables": variables[:-1],
        "domains": domains, 
        "instance": s,
        "simulation": sim
    }

"""Noisy SMK"""
def noise(value, t=.01):
    if np.random.rand() < t:
        return 1-value
    return value

def simulate_suzy_noisy(rule, u, n_var):
    cf = dict(rule)
    s = np.zeros(n_var, dtype=int).tolist()
    ST, BT, SH, BH, BS = range(len(s))
    s[ST] = cf.get(ST, noise(u[0]))
    s[BT] = cf.get(BT, noise(u[1]))
    s[SH] = cf.get(SH, noise(s[ST]))
    s[BH] = cf.get(BH, noise(s[BT] and not s[SH]))
    s[BS] = cf.get(BS, noise(s[BH] or s[SH]))
    return (s, s[-1], sum(s)/len(s))

def get_noisy_suzy_SCM(u=(1,1), lucb_params={}):
    variables = ["ST","BT","SH","BH","BS"]
    domains = ((0,1),)*(len(variables)-1)
    beam_size = 5
    instance = (1,1,1,0)
    sim = lambda rule: simulate_suzy_noisy(rule, u, len(variables))
    return {"domains": domains, "instance":instance, 
            "variables":variables[:-1], 
            "simulation":lambda rules: lucb(sim, rules, **lucb_params)}

def nSMK_model(s, cf, u, n_attacker, nl=0):
    dim_labels = get_nSMK_variables(n_attacker)
    dim2id = dict(zip(dim_labels, range(len(dim_labels))))
    # Set variables from exo source
    for i in range(1,n_attacker+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            s[dim_id] = cf.get(dim_id, u[dim_id])
    # Set depth-1
    for i in range(1,n_attacker+1):
        # Set GP
        s[dim2id[f"GP-U{i}"]] = cf.get(
            dim2id[f"GP-U{i}"], 
            noise(s[dim2id[f"FS-U{i}"]] or s[dim2id[f"FN-U{i}"]], nl)
        )
        # Set GK
        s[dim2id[f"GK-U{i}"]] = cf.get(
            dim2id[f"GK-U{i}"], 
            noise(s[dim2id[f"FF-U{i}"]] or s[dim2id[f"FDB-U{i}"]], nl)
        )
        # Set KMS
        s[dim2id[f"KMS-U{i}"]] = cf.get(
            dim2id[f"KMS-U{i}"], 
            noise(s[dim2id[f"A-U{i}"]] and s[dim2id[f"AD-U{i}"]], nl)
        )
    # Set DK
    for i in range(1,n_attacker+1):
        s[dim2id[f"DK-U{i}"]] = cf.get(
            dim2id[f"DK-U{i}"], 
            noise(s[dim2id[f"GP-U{i}"]] and \
            s[dim2id[f"GK-U{i}"]] and \
            not any([s[dim2id[f"DK-U{j}"]] for j in range(1, i)]), nl)
        )
    # Set SD
    for i in range(1, n_attacker+1):
        s[dim2id[f"SD-U{i}"]] = cf.get(
            dim2id[f"SD-U{i}"], 
            noise(s[dim2id[f"KMS-U{i}"]] and \
            not any([s[dim2id[f"SD-U{j}"]] for j in range(1,i)]), nl)
        )
    # Set global DK
    s[dim2id["DK"]] = cf.get(
        dim2id["DK"], 
        noise(any([s[dim2id[f"DK-U{i}"]] for i in range(1,n_attacker+1)]), nl)
    )
    # Set global SD
    s[dim2id["SD"]] = cf.get(
        dim2id["SD"], 
        noise(any([s[dim2id[f"SD-U{i}"]] for i in range(1,n_attacker+1)]), nl)
    )
    # Set SMK
    s[dim2id["SMK"]] = cf.get(
        dim2id["SMK"], 
        noise(s[dim2id["DK"]] or s[dim2id["SD"]], nl)
    )

def get_nSMK_variables(n_attacker):
    attacker_vars = ["FS", "FN", "FF", "FDB", "A", "AD", 
                     "GP", "GK", "KMS", "DK", "SD"]
    return [
        f"{dim}-U{i}" for dim in attacker_vars for i in range(1,n_attacker+1)
    ] + ["DK", "SD", "SMK"]

def nSMK_simulation(rule, u, n_attacker, nl=0):
    cf = dict(rule)
    variables = get_nSMK_variables(n_attacker)
    s = np.zeros(len(variables), dtype=int).tolist()
    nSMK_model(s, cf, u, n_attacker, nl)
    return (s, s[-1], sum(s)/len(s))

def avg_nSMK_simulation(rules, u, n_attacker, N, nl=0):
    outputs = []
    for rule in rules:
        runs = []
        P, S = 0, 0
        for _ in range(N):
            r, p, s = nSMK_simulation(rule, u, n_attacker, nl)
            runs.append(r)
            P += p
            S += s
        outputs.append((runs, P/N, S/N))
    return outputs

def lucb_nSMK_simulation(rules, u, n_attacker, N, lucb_params, nl=0):
    sim = lambda rule: nSMK_simulation(rule, u, n_attacker, nl)
    return lucb(sim, rules, **lucb_params)
    
def get_nSMK_SCM(n_attacker, u, do_lucb=True, N=100, nl=.05, lucb_params={}):
    variables = get_nSMK_variables(n_attacker)
    s, _, _ = nSMK_simulation([], u, n_attacker, nl=0) # No noise for the instance
    domains = ((0,1),)*(len(variables)-1)
    if do_lucb:
        simulation = lambda rules: lucb_nSMK_simulation(rules, u, n_attacker, N, lucb_params, nl)
    else: 
        simulation = lambda rules: avg_nSMK_simulation(rules, u, n_attacker, N, nl)
    return {"variables": variables[:-1], "domains":domains,
            "simulation": simulation, "instance":s}