import math, numpy as np
from itertools import combinations
from sympy import symbols, Equivalent, to_cnf, Or, And, Not
# from actualcauses import lucb
from actualcauses_local.scm_class import SCM
from actualcauses_local.lucb import lucb
from tqdm import tqdm

"""Base heuristic"""
psi = lambda s: sum(s)-1

"""Rock Throwing"""
suzzy_vars = ["ST","BT","SH","BH","BS"]
suzzy_dag = [[],[],[0],[1,2],[2,3]]
def rock_throwing_model(u: list, e:list[list]):
    """
    u: exogenous values
    e: intervention (variable-value pairs)
    """
    s = [None] * len(suzzy_vars)
    e = dict(e)
    ST, BT, SH, BH, BS = range(len(suzzy_vars))
    s[ST] = e.get(ST, u[0])
    s[BT] = e.get(BT, u[1])
    s[SH] = e.get(SH, s[ST])
    s[BH] = e.get(BH, int(s[BT] and not s[SH]))
    s[BS] = e.get(BS, int(s[BH] or s[SH]))
    return s

scm_suzzy = SCM(
    V=suzzy_vars, 
    U=["st","bt"], 
    D=[(0,1)]*5, 
    F=rock_throwing_model, 
    u=(1,1), 
    psi=psi, 
    dag=suzzy_dag
)

"""Forest Fire"""
# def forest_fire_model(s, V_exo, cf, disjunctive):
#     L, MD, FF = range(len(s))
#     s[L] = cf.get(L, V_exo[0])
#     s[MD] = cf.get(MD, V_exo[1])
#     s[FF] = cf.get(FF, 
#             s[L] or s[MD] if disjunctive 
#             else s[L] and s[MD])

# forest_fire_variables=["L","MD","FF"], 
# forest_fire_exo=(1,1)

"""Rock Throwing Extended"""
# def ext_rock_throwing_model(s, V_exo, cf):
#     ST, BT, W, SH, BH, BS = range(len(s))
#     s[ST] = cf.get(ST, V_exo[0])
#     s[BT] = cf.get(BT, V_exo[1])
#     s[W] = cf.get(W, V_exo[2])
#     s[SH] = cf.get(SH, s[ST] and not s[W])
#     s[BH] = cf.get(BH, s[BT] and not s[SH] and not s[W])
#     s[BS] = cf.get(BS, s[BH] or s[SH])

# ext_rock_throwing_variables=["ST","BT","W","SH","BH","BS"] 
# ext_rock_throwing_exo=(1,1,0)

"""Prisoners"""
# def prisoners_model(s, V_exo, cf):
#     A, B, C, D = range(len(s))
#     s[A] = cf.get(A, V_exo[0])
#     s[B] = cf.get(B, V_exo[1])
#     s[C] = cf.get(C, V_exo[2])
#     s[D] = cf.get(D, (s[A] and s[B]) or s[C])

# prisoners_variables=["A","B","C", "D"]
# prisoners_exo=(1,1,1)

"""Assassin"""
# def assassin_model(s, V_exo, cf, variant):
#     A, B, VS = range(len(s))
#     s[B] = cf.get(B, V_exo[1])
#     s[A] = cf.get(A, V_exo[0] if variant == 1 
#                      else V_exo[0] and s[B])
#     s[VS] = cf.get(VS, not s[A] or s[B])

# assassin_variables=["A","B","VS"]
# assassin_exo=(1,1)

"""Railroad"""
# def railroad_model(s, V_exo, cf):
#     LB, F, RB, A = range(len(s))
#     s[LB] = cf.get(LB, V_exo[0])
#     s[F] = cf.get(F, V_exo[1])
#     s[RB] = cf.get(RB, V_exo[2])
#     s[A] = cf.get(A, 
#                   not (
#                       (s[F] and s[RB]) or (not s[F] and s[LB])
#                   ) 
#                  )

# railroad_variables=["LB","F", "RB","A"]
# railroad_exo=(0,1,0)

"""Abstract Model 1"""
# def first_abstract_model(s, V_exo, cf):
#     A, B, C = range(len(s))
#     s[A] = cf.get(A, V_exo[0])
#     s[B] = cf.get(B, V_exo[1])
#     s[C] = cf.get(C, (s[A] and s[B]) or (not s[A] and not s[B]))

# first_abstract_variables=["A","B","C"]
# first_abstract_exo=(1,1)

"""Abstract Model 2"""
# def second_abstract_model(s, V_exo, cf):
#     A, B, C, D, E, G, H, I = range(len(s))
#     s[A] = cf.get(A, V_exo[0])
#     s[B] = cf.get(B, V_exo[1] and not s[A])
#     s[C] = cf.get(C, s[A] or s[B])
#     s[D] = cf.get(D, s[A])
#     s[E] = cf.get(E, not s[A])
#     s[G] = cf.get(G, not s[C])
#     s[H] = cf.get(H, not s[C] and not s[G])
#     s[I] = cf.get(I, any([s[C], s[D], s[E], s[G], s[H]]))

# second_abstract_variables=["A","B","C","D","E","G","H","I"]
# second_abstract_exo=(1,1)

"""LSP Model"""
# def lsp_model(s, V_exo, cf):
#     dim_labels = [f"X{i}" for i in range(1,42)]
#     dim2id = dict(zip(dim_labels, range(len(dim_labels))))
#     for i in range(1,26):
#         dim = dim2id[f"X{i}"]
#         s[dim] = cf.get(dim, V_exo[dim])
#     equations = [
#         (27, "or", [3,4]),
#         (28, "or", [5,6]),
#         (29, "or", [7,8]),
#         (30, "or", [9,10]),
#         (31, "or", [12,13,14,15,16]),
#         (32, "and", [18,19]),
#         (33, "and", [20,21]),
#         (34, "and", [22,23]),
#         (35, "and", [24,25]),
#         (36, "or", [27,28,29,30]),
#         (37, "and", [31,17]),
#         (38, "and", [1,2]),
#         (39, "and", [36,11]),
#         (40, "or", [37,32,33,34,35]),
#         (41, "or", [38,39,40,26]),
#     ]
#     for target, connector, sources in equations:
#         if connector == "or":
#             value = any([s[dim2id[f"X{i}"]] for i in sources])
#         else:
#             value = all([s[dim2id[f"X{i}"]] for i in sources])
#         dim = dim2id[f"X{target}"]
#         s[dim] = cf.get(dim, value)

# lsp_variables=[f"X{i}" for i in range(1,42)]

"""Binary trees"""
# def get_n_leaves(depth):
#     n_nodes = 2**depth-1
#     return math.ceil((n_nodes + 1) / 2)
    
# def binary_tree_model(s, V_exo, cf, depth):
#     n_nodes = 2**depth-1
#     n_leaves = math.ceil((n_nodes + 1) / 2)
#     for i in range(n_leaves, n_nodes):
#         s[i] = cf.get(i, V_exo[n_nodes-i])
#     for i in range(n_nodes-n_leaves-1, -1, -1):
#         s[i] = cf.get(i, s[2*i+1] or s[2*i+2])

"""Steal Master Key"""
smk_base_vars_exo = "fs fn ff fdb a ad".split()
smk_base_vars_users = "FS FN FF FDB A AD GP GK KMS DK SD".split()

def get_SMK_V(n):
    return [
        f"{dim}-U{i}" for dim in smk_base_vars_users for i in range(1,n+1)
    ] + ["DK", "SD", "SMK"]

def get_SMK_U(n):
    return [
        f"{dim}-U{i}" for dim in smk_base_vars_exo for i in range(1,n+1)
    ]

def SMK_model(u:list, e:list[list], n:int):
    """
    u: exogenous values
    e: intervention (variable-value pairs)
    n: number of attackers
    """
    V = get_SMK_V(n)
    s = [None] * len(V)
    dim2id = dict(zip(V, range(len(V))))
    e = dict(e)
    # Set variables from exo source
    for i in range(1,n+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            s[dim_id] = e.get(dim_id, u[dim_id])
    # Set depth-1
    for i in range(1,n+1):
        # Set GP
        s[dim2id[f"GP-U{i}"]] = e.get(
            dim2id[f"GP-U{i}"], 
            int(s[dim2id[f"FS-U{i}"]] or s[dim2id[f"FN-U{i}"]]) 
        )
        # Set GK
        s[dim2id[f"GK-U{i}"]] = e.get(
            dim2id[f"GK-U{i}"], 
            int(s[dim2id[f"FF-U{i}"]] or s[dim2id[f"FDB-U{i}"]])
        )
        # Set KMS
        s[dim2id[f"KMS-U{i}"]] = e.get(
            dim2id[f"KMS-U{i}"], 
            int(s[dim2id[f"A-U{i}"]] and s[dim2id[f"AD-U{i}"]])
        )
    # Set DK
    for i in range(1,n+1):
        s[dim2id[f"DK-U{i}"]] = e.get(
            dim2id[f"DK-U{i}"], 
            int(s[dim2id[f"GP-U{i}"]] and \
            s[dim2id[f"GK-U{i}"]] and \
            not any([s[dim2id[f"DK-U{j}"]] for j in range(1, i)]))
        )
    # Set SD
    for i in range(1, n+1):
        s[dim2id[f"SD-U{i}"]] = e.get(
            dim2id[f"SD-U{i}"], 
            int(s[dim2id[f"KMS-U{i}"]] and \
            not any([s[dim2id[f"SD-U{j}"]] for j in range(1,i)]))
        )
    # Set global DK
    s[dim2id["DK"]] = e.get(
        dim2id["DK"], 
        int(any([s[dim2id[f"DK-U{i}"]] for i in range(1,n+1)]))
    )
    # Set global SD
    s[dim2id["SD"]] = e.get(
        dim2id["SD"], 
        int(any([s[dim2id[f"SD-U{i}"]] for i in range(1,n+1)]))
    )
    # Set SMK
    s[dim2id["SMK"]] = e.get(
        dim2id["SMK"], 
        int(s[dim2id["DK"]] or s[dim2id["SD"]])
    )

    return s

def get_SMK_DAG(n:int):
    """
    Create the DAG for the SMK scenario with n attackers.
        n[int]: number of attackers
    """
    V = get_SMK_V(n)
    dag = [[] for v in V]
    var2int = {v: i for i,v in enumerate(V)}
    for n in range(1,n+1):
        dag[var2int[f'GP-U{n}']] += [var2int[f'FS-U{n}'], var2int[f'FN-U{n}']]
        dag[var2int[f'GK-U{n}']] += [var2int[f'FF-U{n}'], var2int[f'FDB-U{n}']]
        dag[var2int[f'KMS-U{n}']] += [var2int[f'A-U{n}'], var2int[f'AD-U{n}']]
        dag[var2int[f'DK-U{n}']] += [var2int[f'GP-U{n}'], var2int[f'GK-U{n}']] + [var2int[f'DK-U{j}'] for j in range(1, n)]
        dag[var2int[f'SD-U{n}']] += [var2int[f'KMS-U{n}']] + [var2int[f'SD-U{j}'] for j in range(1, n)]
        dag[var2int['DK']] += [var2int[f'DK-U{n}']]
        dag[var2int['SD']] += [var2int[f'SD-U{n}']]
    dag[var2int["SMK"]] += [var2int["SD"], var2int["DK"]]
    return dag

def get_SMK_SCM(n:int, u:list):
    """
    Creates an SCM for the SMK scenario with n attackers and context u.
        n[int]: number of attackers
    """
    V = get_SMK_V(n)
    return SCM(
        V=V,
        U=get_SMK_U(n),
        D=[(0,1)] * len(V),
        F=lambda u,e: SMK_model(u,e,n),
        u=u,
        psi=psi,
        dag=get_SMK_DAG(n)
    )

def get_sympy_SMK(n):
    leaf_vars = ["FS", "FN", "FF", "FDB", "A", "AD"]
    node_vars = ["GP", "GK", "KMS", "DK", "SD"]
    
    exo_var_labels = [
        f"{dim.lower()}-u{i}" \
    for dim in leaf_vars \
    for i in range(1,n+1)
    ]
    
    var_labels = [
            f"{dim}-U{i}" \
        for dim in leaf_vars + node_vars \
        for i in range(1,n+1)
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
    for i in range(1,n+1):
        GP, FS, FN, GK, FF, FDB, KMS, A, AD = [
            variables[label2id[f"{var_name}-U{i}"]] for \
            var_name in "GP, FS, FN, GK, FF, FDB, KMS, A, AD".split(", ")
        ]
        
        equations[GK] = FF | FDB
        equations[KMS] = A & AD
        equations[GP] = FS | FN
        
    # Set DK
    for i in range(1,n+1):
        equations[variables[label2id[f"DK-U{i}"]]] = \
        variables[label2id[f"GP-U{i}"]] & variables[label2id[f"GK-U{i}"]] &\
        And(*[~variables[label2id[f"DK-U{j}"]] for j in range(1, i)])
    
    # Set SD
    for i in range(1,n+1):
        equations[variables[label2id[f"SD-U{i}"]]] = \
        variables[label2id[f"KMS-U{i}"]] &\
        And(*[~variables[label2id[f"SD-U{j}"]] for j in range(1, i)])
    
    # Set global DK
    equations[variables[label2id["DK"]]] = \
    Or(*[variables[label2id[f"DK-U{i}"]] for i in range(1, n+1)])
    
    
    # Set global SD
    equations[variables[label2id["SD"]]] = \
    Or(*[variables[label2id[f"SD-U{i}"]] for i in range(1, n+1)])
    
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

def mSMK_model(u, e, n):
    s = [None] * len(mSMK_variables)
    e = dict(e)
    
    dim2id = dict(zip(mSMK_variables, range(len(mSMK_variables))))
            
    # Set from exo
    labels = ("FS", "FN", "FF", "FDB", "A", "AD")
    for i, label in enumerate(labels):
        dim_id = dim2id[label]
        s[dim_id] = e.get(dim_id, tuple([j for j in range(n) if u[i * n + j]]))
    
    # Set KMS
    s[dim2id["KMS"]] = e.get(dim2id["KMS"], tuple(set(s[dim2id["A"]]) & set(s[dim2id["AD"]])))
    
    # Set SD
    s[dim2id["SD"]] = e.get(dim2id["SD"], min(s[dim2id["KMS"]]) if s[dim2id["KMS"]] else -1)

    # Set GK 
    s[dim2id["GK"]] = e.get(dim2id["GK"], tuple(set(s[dim2id["FF"]]) | set(s[dim2id["FDB"]])))
    
    # Set GP
    s[dim2id["GP"]] = e.get(dim2id["GP"], tuple(set(s[dim2id["FS"]]) | set(s[dim2id["FN"]])))

    # Set DK
    children = tuple(set(s[dim2id["GP"]]) & set(s[dim2id["GK"]]))
    s[dim2id["DK"]] = e.get(dim2id["DK"], min(children) if children else -1)
    
    # Set SMK
    s[dim2id["SMK"]] = e.get(dim2id["SMK"], int(s[dim2id["DK"]] > -1 or s[dim2id["SD"]] > -1))
    return s

def get_mSMK_domains(n):
    return [
        generate_subsets(n) for var in ("FS", "FN", "FF", "FDB", "A", "AD", "GP", "GK", "KMS") 
    ] + [
        [-1] + list(range(n)),
        [-1] + list(range(n)),
        [0,1]
    ]

def mSMK_heuristic(s):
    dim2id = dict(zip(mSMK_variables, range(len(mSMK_variables))))
    return (
        sum([len(s[dim2id[dim]]) for dim in ("FS", "FN", "FF", "FDB", "A", "AD", "GP", "GK", "KMS")]) + 
        s[dim2id["DK"]] + s[dim2id["SD"]]
    )

def get_mSMK_DAG():
    dag = [[] for v in mSMK_variables]
    var2int = {v: i for i,v in enumerate(mSMK_variables)}
    dag[var2int['GP']] += [var2int['FS'], var2int['FN']]
    dag[var2int['GK']] += [var2int['FF'], var2int['FDB']]
    dag[var2int['KMS']] += [var2int['A'], var2int['AD']]
    dag[var2int['DK']] += [var2int['GP'], var2int['GK']]
    dag[var2int['SD']] += [var2int['KMS']]
    dag[var2int['SMK']] += [var2int['DK'], var2int['SD']]
    return dag

def get_mSMK_SCM(n, u):
    return SCM(
        V=mSMK_variables,
        U=smk_base_vars_exo,
        D=get_mSMK_domains(n),
        F=lambda u,e: mSMK_model(u,e,n),
        u=u,
        psi=mSMK_heuristic,
        dag=get_mSMK_DAG()
    )

"""Black Box SMK"""
    
def get_bbSMK_V(n):
    return [f"{label}-U{i}" for label in ("FS", "FN", "FF", "FDB", "A", "AD") for i in range(1,n+1)] + ["SMK"]

def bbSMK_model(u, e, n):
    V = get_bbSMK_V(n)
    e = dict(e)
    dim2id = dict(zip(V, range(len(V))))
    s = [None] * len(V)
            
    # Set from exo
    for i in range(1,n+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            s[dim_id] = e.get(dim_id, u[dim_id])
    GP = [s[dim2id[f"FS-U{i}"]] or s[dim2id[f"FN-U{i}"]] for i in range(1, n+1)]
    GK = [s[dim2id[f"FF-U{i}"]] or s[dim2id[f"FDB-U{i}"]] for i in range(1, n+1)]
    KMS = [s[dim2id[f"A-U{i}"]] and s[dim2id[f"AD-U{i}"]] for i in range(1, n+1)]
    DK = any([gp and gk for gp, gk in zip(GP,GK)])
    SD = any(KMS)
        
    # Set SMK
    s[dim2id["SMK"]] = e.get(
        dim2id["SMK"], 
        int(DK or SD)
    )
    return s

def get_bbSMK_SCM(n, u):
    return SCM(
        V=get_bbSMK_V(n),
        U=get_SMK_U(n),
        D=[(0,1)] * (6 * n + 1),
        F=lambda u,e: bbSMK_model(u,e,n),
        u=u,
        psi=psi,
        dag=suzzy_dag
    )

"""Noisy SMK"""
def noise(value, t=.01):
    if np.random.rand() < t:
        return int(1-value)
    return int(value)

def suzy_noisy_model(u, e, t):
    e = dict(e)
    s = np.zeros(len(suzzy_vars), dtype=int).tolist()
    ST, BT, SH, BH, BS = range(len(s))
    s[ST] = e.get(ST, noise(u[0], t))
    s[BT] = e.get(BT, noise(u[1], t))
    s[SH] = e.get(SH, noise(s[ST], t))
    s[BH] = e.get(BH, noise(s[BT] and not s[SH], t))
    s[BS] = e.get(BS, noise(s[BH] or s[SH], t))
    return s

def get_noisy_suzzy_SCM(u, t, lucb_params):
    F = lambda u,e: suzy_noisy_model(u,e,t)
    
    def evaluator(e):
        s = suzy_noisy_model(u,e,t)
        return s[-1], psi(s)
        
    return SCM(
        V=suzzy_vars,
        U=("st", "bt"),
        D=[(0,1)] * len(suzzy_vars),
        F=F,
        u=(1,1),
        psi=psi,
        dag=None,
        sim=lambda E: lucb(evaluator, E, **lucb_params)
    )

def nSMK_model(u, e, n, t):
    V = get_SMK_V(n)
    e = dict(e)
    dim2id = dict(zip(V, range(len(V))))
    s = [None] * len(V)
    # Set variables from exo source
    for i in range(1,n+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            s[dim_id] = e.get(dim_id, u[dim_id])
    # Set depth-1
    for i in range(1,n+1):
        # Set GP
        s[dim2id[f"GP-U{i}"]] = e.get(
            dim2id[f"GP-U{i}"], 
            noise(s[dim2id[f"FS-U{i}"]] or s[dim2id[f"FN-U{i}"]], t)
        )
        # Set GK
        s[dim2id[f"GK-U{i}"]] = e.get(
            dim2id[f"GK-U{i}"], 
            noise(s[dim2id[f"FF-U{i}"]] or s[dim2id[f"FDB-U{i}"]], t)
        )
        # Set KMS
        s[dim2id[f"KMS-U{i}"]] = e.get(
            dim2id[f"KMS-U{i}"], 
            noise(s[dim2id[f"A-U{i}"]] and s[dim2id[f"AD-U{i}"]], t)
        )
    # Set DK
    for i in range(1,n+1):
        s[dim2id[f"DK-U{i}"]] = e.get(
            dim2id[f"DK-U{i}"], 
            noise(s[dim2id[f"GP-U{i}"]] and \
            s[dim2id[f"GK-U{i}"]] and \
            not any([s[dim2id[f"DK-U{j}"]] for j in range(1, i)]), t)
        )
    # Set SD
    for i in range(1, n+1):
        s[dim2id[f"SD-U{i}"]] = e.get(
            dim2id[f"SD-U{i}"], 
            noise(s[dim2id[f"KMS-U{i}"]] and \
            not any([s[dim2id[f"SD-U{j}"]] for j in range(1,i)]), t)
        )
    # Set global DK
    s[dim2id["DK"]] = e.get(
        dim2id["DK"], 
        noise(any([s[dim2id[f"DK-U{i}"]] for i in range(1,n+1)]), t)
    )
    # Set global SD
    s[dim2id["SD"]] = e.get(
        dim2id["SD"], 
        noise(any([s[dim2id[f"SD-U{i}"]] for i in range(1,n+1)]), t)
    )
    # Set SMK
    s[dim2id["SMK"]] = e.get(
        dim2id["SMK"], 
        noise(s[dim2id["DK"]] or s[dim2id["SD"]], t)
    )
    return s

def avg_nSMK_model(u, e, n, t, N):
    V = get_SMK_V(n)
    S = np.zeros(len(V))
    for _ in range(N):
        s = nSMK_model(u, e, n, t)
        S += s
    return (S / N).tolist()

def get_avg_nSMK_SCM(n, u, N, nl):
    V = get_SMK_V(n)
    v = nSMK_model(u, [], n, t=0) # No noise for the instance
    t = nl/len(V) # Compute the noise threshold with the noise level
    return SCM(
        V=V,
        U=get_SMK_U(n),
        D=(0,1),
        F=lambda u,e: avg_nSMK_model(u, e, n, t, N),
        u=u,
        psi=psi,
        dag=get_SMK_DAG(n),
        v=v,
    )
    
def get_lucb_nSMK_SCM(n, u, nl, lucb_params=None):
    V = get_SMK_V(n)
    v = nSMK_model(u, [], n, t=0) # No noise for the instance
    t = nl/len(V) # Compute the noise threshold with the noise level
    
    def lucb_evaluator(e):
        s = nSMK_model(u, e, n, t)
        return s[-1], psi(s)/len(s)
        
    return SCM(
        V=V,
        U=get_SMK_U(n),
        D=(0,1),
        F=lambda u,e: nSMK_model(u,e,n,t),
        u=u,
        psi=psi,
        dag=get_SMK_DAG(n),
        sim=lambda E: lucb(lucb_evaluator, E, **lucb_params),
        v=v,
    )