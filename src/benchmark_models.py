import math, numpy as np
from utils import generate_subsets, elementwise_any, elementwise_max
from sympy import symbols, Equivalent, to_cnf, Or, And, Not
# from actualcauses import lucb
from actualcauses_local.scm_class import SCM
from actualcauses_local.lucb import lucb
from tqdm import tqdm
from collections import defaultdict

"""Base heuristic"""
psi = lambda s: sum(s)-1

"""Rock Throwing"""
suzzy_vars = ["ST","BT","SH","BH","BS"]
suzzy_dag = {"ST":[],"BT":[],"SH":["ST"],"BH":["BT","SH"],"BS":["BH","SH"]}
def rock_throwing_model(u: list, e:list[list]):
    e = dict(e)
    s = [None] * len(suzzy_vars)
    ST, BT, SH, BH, BS = range(len(suzzy_vars))
    s[ST] = e.get("ST", u[0])
    s[BT] = e.get("BT", u[1])
    s[SH] = e.get("SH", s[ST])
    s[BH] = e.get("BH", int(s[BT] and not s[SH]))
    s[BS] = e.get("BS", int(s[BH] or s[SH]))
    return s

scm_suzzy = SCM(V=suzzy_vars, U=["st","bt"], D=[(0,1)]*5, F=rock_throwing_model, 
                u=(1,1), psi=psi, dag=suzzy_dag)

"""Examples from the ISI corectness proof"""
def or_model(u, e):
    s = [None] * 4
    e = dict(e)
    s[0] = e.get("X", u[0])
    s[1] = e.get("A", s[0])
    s[2] = e.get("B", 1 and not s[1])
    s[3] = (s[1] or s[2])# or (not s[1] and not s[2])
    return s
    
or_scm = SCM(V=["X", "A", "B", "T"],U=["x"],u=[1],D=(0,1),F=or_model,
             psi=lambda x: 1,dag={"X":[], "A":["X"], "B":["A"], "T":["A", "B"]})

def xnor_model(u, e):
    s = [None] * 4
    e = dict(e)
    s[0] = e.get("X", u[0])
    s[1] = e.get("A", s[0])
    s[2] = e.get("B", not s[0])
    s[3] = s[1] != s[2]
    return s
    
xnor_scm = SCM(V=["X", "A", "B", "T"],U=["x"],u=[1],D=(0,1),
               F=xnor_model,psi=lambda x: 1,dag={"X":[], "A":["X"], "B":["X"], "T":["A", "B"]})

def chain_model(u, e):
    s = [None] * 6
    e = dict(e)
    s[0] = e.get("X", u[0])
    s[1] = e.get("J", s[0])
    s[2] = e.get("H", s[1])
    if s[2] and not s[1]: # H and not J
        g = "gx"
    elif (s[2] and s[1]) or (not s[2] and s[1]): # (H and J) or (not H and J)
        g = "g*"
    elif not s[2] and not s[1]: # not H and not J
        g = "g'"
    s[3] = e.get("G", g)
    s[4] = e.get("B", (s[3] != "gx") or s[0])
    s[5] = s[4]
    return s

chain_scm = SCM(V=["X", "J", "H", "G", "B", "T"],U=["x"],u=[1],
                D=[(0,1),(0,1),(0,1),("g'","g*", "gx"), (0,1), (0,1)],
                F=chain_model,psi=lambda s: 1,
                dag={"T":["B"], "B":["X","J","G"], "G": ["H", "J"], "H": ["J"], "J": ["X"], "X": []})

def split_model(u, e):
    s = [None] * 7
    e = dict(e)
    s[0] = e.get("X", u[0])
    s[1] = e.get("G", u[1])
    s[2] = e.get("Y", u[2])
    s[3] = e.get("H", u[3])
    if s[1]: a = "g1"
    else: a = "x1" if s[0] else "x0"
    s[4] = e.get("A", a)
    if s[3]: b = "h1"
    else: b = "y1" if s[2] else "y0"
    s[5] = e.get("B", b)
    t = not (s[4] == "x0" or s[5] == "y0" or (s[4]=="g1" and s[5]=="h1"))
    s[6] = e.get("T", t)
    return s

split_scm = SCM(V=["X", "G", "Y", "H", "A", "B", "T"],U=["x", "g", "y", "h"],
                D=[(0,1), (0,1), (0,1), (0,1), ("x0","x1","g0","g1"),("y0","y1","h0","h1"),(0,1)],
                u=(1,0,1,0),F=split_model,psi=lambda s:1,
                dag={"X":[],"G":[],"Y":[],"H":[],"A":["X","G"],"B":["Y","H"],"T":["A","B"]})

"""Steal Master Key"""
smk_base_vars_exo = "fs fn ff fdb a ad".split()
smk_base_vars_users = "FS FN FF FDB A AD GP GK KMS DK SD".split()

def get_SMK_V(n):
    return [f"{dim}-U{i}" for dim in smk_base_vars_users for i in range(1,n+1)] + ["DK", "SD", "SMK"]

def get_SMK_U(n):
    return [
        f"{dim}-U{i}" for dim in smk_base_vars_exo for i in range(1,n+1)
    ]

def set_value_vect(s, var, dim2id, F_value, e, t=0):
    """
    E is the interventions in the form: var -> list[(horizontal slice, value)]
    """
    s[:,dim2id[var]] = F_value
    if t > 0:
        ids = np.random.rand(s.shape[0]) < t
        s[ids,dim2id[var]] = 1 - s[ids,dim2id[var]]
    if var in e:
        for h_slice, value in e[var]:
            s[h_slice,dim2id[var]] = value
    
def vectorized_SMK_model(u, E, n, N=1, t=0):
    formated_E = defaultdict(lambda: [])
    for i, e in enumerate(E):
        for var, value in e:
            formated_E[var].append((slice(i*N,(i+1)*N),value))
    V = get_SMK_V(n)
    s = np.zeros((len(E)*N, len(V)), dtype=bool)
    dim2id = dict(zip(V, range(len(V))))
    for i in range(1,n+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            set_value_vect(s, f"{dim}-U{i}", dim2id, u[dim_id], formated_E)# Set depth-1
    for i in range(1,n+1):
        set_value_vect(s, f"GP-U{i}", dim2id, s[:,dim2id[f"FS-U{i}"]] | s[:,dim2id[f"FN-U{i}"]], formated_E, t)
        set_value_vect(s, f"GK-U{i}", dim2id, s[:,dim2id[f"FF-U{i}"]] | s[:,dim2id[f"FDB-U{i}"]], formated_E, t)
        set_value_vect(s, f"KMS-U{i}", dim2id, s[:,dim2id[f"A-U{i}"]] & s[:,dim2id[f"AD-U{i}"]], formated_E, t)
    for i in range(1,n+1):
        block_DK = elementwise_any([s[:,dim2id[f"DK-U{j}"]] for j in range(1, i)])
        set_value_vect(s, f"DK-U{i}", dim2id, s[:,dim2id[f"GP-U{i}"]] & s[:,dim2id[f"GK-U{i}"]] & ~block_DK, formated_E, t)
        block_SD = elementwise_any([s[:,dim2id[f"SD-U{j}"]] for j in range(1, i)])
        set_value_vect(s, f"SD-U{i}", dim2id, s[:,dim2id[f"KMS-U{i}"]] & ~block_SD, formated_E)
    set_value_vect(s, "DK", dim2id, elementwise_any([s[:,dim2id[f"DK-U{j}"]] for j in range(1, n+1)]), formated_E, t)
    set_value_vect(s, "SD", dim2id, elementwise_any([s[:,dim2id[f"SD-U{j}"]] for j in range(1, n+1)]), formated_E, t)
    set_value_vect(s, "SMK", dim2id, s[:,dim2id["DK"]] | s[:,dim2id["SD"]], formated_E, t)
    return s
    
def SMK_model(u:list, e:list[list], n:int):
    V = get_SMK_V(n)
    s = [None] * len(V)
    dim2id = dict(zip(V, range(len(V))))
    e = dict(e)
    # Set variables from exo source
    for i in range(1,n+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            s[dim_id] = e.get(f"{dim}-U{i}", u[dim_id])
    # Set depth-1
    for i in range(1,n+1):
        s[dim2id[f"GP-U{i}"]] = e.get(f"GP-U{i}", int(s[dim2id[f"FS-U{i}"]] or s[dim2id[f"FN-U{i}"]]))
        s[dim2id[f"GK-U{i}"]] = e.get(f"GK-U{i}", int(s[dim2id[f"FF-U{i}"]] or s[dim2id[f"FDB-U{i}"]]))
        # Set KMS
        s[dim2id[f"KMS-U{i}"]] = e.get(f"KMS-U{i}", int(s[dim2id[f"A-U{i}"]] and s[dim2id[f"AD-U{i}"]]))
    # Set depth-2
    for i in range(1,n+1):
        s[dim2id[f"DK-U{i}"]] = e.get(f"DK-U{i}", 
            int(s[dim2id[f"GP-U{i}"]] and s[dim2id[f"GK-U{i}"]] and not any([s[dim2id[f"DK-U{j}"]] for j in range(1, i)])))
    for i in range(1, n+1):
        s[dim2id[f"SD-U{i}"]] = e.get(f"SD-U{i}", 
            int(s[dim2id[f"KMS-U{i}"]] and not any([s[dim2id[f"SD-U{j}"]] for j in range(1,i)])))
    # Set depth -3
    s[dim2id["DK"]] = e.get("DK", int(any([s[dim2id[f"DK-U{i}"]] for i in range(1,n+1)])))
    s[dim2id["SD"]] = e.get("SD", int(any([s[dim2id[f"SD-U{i}"]] for i in range(1,n+1)])))
    s[dim2id["SMK"]] = e.get("SMK", int(s[dim2id["DK"]] or s[dim2id["SD"]]))
    return s

def get_SMK_DAG(n:int):
    """
    Create the DAG for the SMK scenario with n attackers.
        n[int]: number of attackers
    """
    V = get_SMK_V(n)
    dag = {v:[] for v in V}
    for n in range(1,n+1):
        dag[f'GP-U{n}'] += [f'FS-U{n}', f'FN-U{n}']
        dag[f'GK-U{n}'] += [f'FF-U{n}', f'FDB-U{n}']
        dag[f'KMS-U{n}'] += [f'A-U{n}', f'AD-U{n}']
        dag[f'DK-U{n}'] += [f'GP-U{n}', f'GK-U{n}'] + [f'DK-U{j}' for j in range(1, n)]
        dag[f'SD-U{n}'] += [f'KMS-U{n}'] + [f'SD-U{j}' for j in range(1, n)]
        dag['DK'] += [f'DK-U{n}']
        dag['SD'] += [f'SD-U{n}']
    dag["SMK"] += ["SD", "DK"]
    return dag

def get_SMK_SCM(n:int, u:list):
    """
    Creates an SCM for the SMK scenario with n attackers and context u.
        n[int]: number of attackers
    """
    V = get_SMK_V(n)
    def vectorized_simulations(E):
        out = []
        sub_N = 100_000
        for i in range(0, len(E), sub_N):
            sub_E = E[i*sub_N:(i+1)*sub_N]
            S = vectorized_SMK_model(u, sub_E, n)
            for s in S:
                out.append((s, int(s[-1]), psi(s)))
        return out
        
    return SCM(V=V,U=get_SMK_U(n),D=[(0,1)] * len(V), F=lambda u,e: SMK_model(u,e,n),
               u=u,psi=psi,dag=get_SMK_DAG(n), sim=vectorized_simulations)

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
mSMK_variables = smk_base_vars_users + ["SMK"]

def mSMK_model(u:list, e:list[list], n:int):
    V = get_SMK_V(n)
    s = [None] * len(V)
    dim2id = dict(zip(V, range(len(V))))
    e = dict(e)
    # Set variables from exo source
    for i in range(1,n+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            s[dim_id] = e.get(f"{dim}-U{i}", u[dim_id])
    # print(s)
    # Set depth-1
    for i in range(1,n+1):
        s[dim2id[f"GP-U{i}"]] = e.get(f"GP-U{i}", s[dim2id[f"FS-U{i}"]] + s[dim2id[f"FN-U{i}"]])
        s[dim2id[f"GK-U{i}"]] = e.get(f"GK-U{i}", s[dim2id[f"FF-U{i}"]] + s[dim2id[f"FDB-U{i}"]])
        # Set KMS
        s[dim2id[f"KMS-U{i}"]] = e.get(f"KMS-U{i}", s[dim2id[f"A-U{i}"]] * s[dim2id[f"AD-U{i}"]])
    # print(s)
    # Set depth-2
    for i in range(1,n+1):
        block = max([s[dim2id[f"DK-U{j}"]] for j in range(1, i)]) if i > 1 else 0
        s[dim2id[f"DK-U{i}"]] = e.get(f"DK-U{i}", 
            max(0, s[dim2id[f"GP-U{i}"]] * s[dim2id[f"GK-U{i}"]] - block))
    # print(s)
    for i in range(1, n+1):
        block = max([s[dim2id[f"SD-U{j}"]] for j in range(1, i)]) if i > 1 else 0
        s[dim2id[f"SD-U{i}"]] = e.get(f"SD-U{i}", 
            max(0, s[dim2id[f"KMS-U{i}"]] - block))
    # print(s)
    # Set depth -3
    s[dim2id["DK"]] = e.get("DK", max([s[dim2id[f"DK-U{i}"]] for i in range(1,n+1)]))
    # print(s)
    s[dim2id["SD"]] = e.get("SD", max([s[dim2id[f"SD-U{i}"]] for i in range(1,n+1)]))
    # print(s, s[dim2id["DK"]]>0, s[dim2id["SD"]>0)
    s[dim2id["SMK"]] = e.get("SMK", s[dim2id["DK"]]>0 or s[dim2id["SD"]]>0)
    # print(s)
    return s

def vectorized_mSMK_model(u, E, n, N=1, t=0):
    formated_E = defaultdict(lambda: [])
    for i, e in enumerate(E):
        for var, value in e:
            formated_E[var].append((slice(i*N,(i+1)*N),value))
    V = get_SMK_V(n)
    s = np.zeros((len(E)*N, len(V)), dtype=int)
    dim2id = dict(zip(V, range(len(V))))
    for i in range(1,n+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            set_value_vect(s, f"{dim}-U{i}", dim2id, u[dim_id], formated_E)
    for i in range(1,n+1):
        set_value_vect(s, f"GP-U{i}", dim2id, s[:,dim2id[f"FS-U{i}"]] + s[:,dim2id[f"FN-U{i}"]], formated_E, t)
        set_value_vect(s, f"GK-U{i}", dim2id, s[:,dim2id[f"FF-U{i}"]] + s[:,dim2id[f"FDB-U{i}"]], formated_E, t)
        set_value_vect(s, f"KMS-U{i}", dim2id, s[:,dim2id[f"A-U{i}"]] * s[:,dim2id[f"AD-U{i}"]], formated_E, t)
    for i in range(1,n+1):
        block_DK = elementwise_max([s[:,dim2id[f"DK-U{j}"]] for j in range(1, i)])
        set_value_vect(s, f"DK-U{i}", dim2id, 
                       np.maximum(0, s[:,dim2id[f"GP-U{i}"]] * s[:,dim2id[f"GK-U{i}"]] - block_DK), 
                                  formated_E, t)
        block_SD = elementwise_max([s[:,dim2id[f"SD-U{j}"]] for j in range(1, i)])
        set_value_vect(s, f"SD-U{i}", dim2id, np.maximum(0, s[:,dim2id[f"KMS-U{i}"]] - block_SD), formated_E)
    set_value_vect(s, "DK", dim2id, elementwise_max([s[:,dim2id[f"DK-U{j}"]] for j in range(1, n+1)]), formated_E, t)
    set_value_vect(s, "SD", dim2id, elementwise_max([s[:,dim2id[f"SD-U{j}"]] for j in range(1, n+1)]), formated_E, t)
    set_value_vect(s, "SMK", dim2id, 
                   np.logical_or(s[:,dim2id["DK"]]>0, s[:,dim2id["SD"]]>0), 
                   formated_E, t)
    return s

def get_mSMK_domains(n):
    # FS, FN, FF, FDB, A, AD / GP, GK / KMS / DKu / SDu / DK / SD, SMK
    return 6 * n * [(0,1)] + 2 * n * [(0,1,2)] + n * [(0,1)] + n * [(0,1,2,3,4)] + n * [(0,1)] + [(0,1,2,3,4)] + 2 * [(0,1)]


def get_mSMK_SCM(n, u):
    V = get_SMK_V(n)
    
    def vectorized_simulations(E):
        out = []
        sub_N = 100_000
        for i in range(0, len(E), sub_N):
            sub_E = E[i*sub_N:(i+1)*sub_N]
            S = vectorized_mSMK_model(u, sub_E, n)
            for s in S:
                out.append((s, int(s[-1]), psi(s)))
        return out
        
    return SCM(V=get_SMK_V(n),U=get_SMK_U(n),
               D=get_mSMK_domains(n),
        F=lambda u,e: mSMK_model(u,e,n),u=u,psi=psi,dag=get_SMK_DAG(n),
              sim=vectorized_simulations)

# def mSMK_model(u, e, n):
#     s = [None] * len(mSMK_variables)
#     e = dict(e)
#     dim2id = dict(zip(mSMK_variables, range(len(mSMK_variables))))
#     labels = ("FS", "FN", "FF", "FDB", "A", "AD")
#     for i, label in enumerate(labels):
#         dim_id = dim2id[label]
#         s[dim_id] = e.get(label, tuple([j for j in range(n) if u[i * n + j]]))
#     s[dim2id["KMS"]] = e.get("KMS", tuple(set(s[dim2id["A"]]) & set(s[dim2id["AD"]])))
#     s[dim2id["SD"]] = e.get("SD", min(s[dim2id["KMS"]]) if s[dim2id["KMS"]] else -1)
#     s[dim2id["GK"]] = e.get("GK", tuple(set(s[dim2id["FF"]]) | set(s[dim2id["FDB"]])))
#     s[dim2id["GP"]] = e.get("GP", tuple(set(s[dim2id["FS"]]) | set(s[dim2id["FN"]])))
#     children = tuple(set(s[dim2id["GP"]]) & set(s[dim2id["GK"]]))
#     s[dim2id["DK"]] = e.get("DK", min(children) if children else -1)
#     s[dim2id["SMK"]] = e.get("SMK", int(s[dim2id["DK"]] > -1 or s[dim2id["SD"]] > -1))
#     return s

# def get_mSMK_domains(n):
#     return [generate_subsets(n) for var in mSMK_variables[:-3]] + [[-1]+list(range(n))] * 2 + [[0,1]]

# def mSMK_heuristic(s):
#     dim2id = dict(zip(mSMK_variables, range(len(mSMK_variables))))
#     return sum([len(s[dim2id[dim]]) for dim in smk_base_vars_users[:-2]]) + s[dim2id["DK"]] + s[dim2id["SD"]]
    
# mSMK_DAG = {"FS":[], "FN":[], "FF":[], "FDB":[], "A":[], "AD": [],
#             "GP": ['FS', 'FN'], "GK":['FF', 'FDB'], "KMS":['A', 'AD'], 
#             "DK": ['GP', 'GK'], "SD": ['KMS'], "SMK": ['DK', 'SD']}

# def get_mSMK_SCM(n, u):
#     return SCM(V=mSMK_variables,U=get_SMK_U(n),
#                D=get_mSMK_domains(n),
#         F=lambda u,e: mSMK_model(u,e,n),u=u,psi=mSMK_heuristic,dag=mSMK_DAG)

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
            s[dim_id] = e.get(f"{dim}-U{i}", u[dim_id])
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
    return SCM(V=get_bbSMK_V(n),U=get_SMK_U(n),D=[(0,1)] * (6 * n + 1),
               F=lambda u,e: bbSMK_model(u,e,n),u=u,psi=psi,dag=None)

"""Noisy SMK"""
def noise(value, t=.01):
    if np.random.rand() < t:
        return int(1-value)
    return int(value)

def suzy_noisy_model(u, e, t):
    e = dict(e)
    s = np.zeros(len(suzzy_vars), dtype=int).tolist()
    ST, BT, SH, BH, BS = range(len(s))
    s[ST] = e.get("ST", noise(u[0], t))
    s[BT] = e.get("BT", noise(u[1], t))
    s[SH] = e.get("SH", noise(s[ST], t))
    s[BH] = e.get("BH", noise(s[BT] and not s[SH], t))
    s[BS] = e.get("BS", noise(s[BH] or s[SH], t))
    return s

def get_noisy_suzzy_SCM(u, t, lucb_params):
    F = lambda u,e: suzy_noisy_model(u,e,t)
    
    def evaluator(E, N):
        out = []
        for e in E:
            out.append([])
            for _ in range(N):
                s = suzy_noisy_model(u,e,t)
                out[-1].append([s[-1], psi(s)/len(s)])
        
        return np.array(out)
        
    return SCM(V=suzzy_vars,U=("st", "bt"),D=[(0,1)] * len(suzzy_vars),
               F=F,u=(1,1),psi=psi,dag=None,
               sim=lambda E: lucb(evaluator, E, **lucb_params))

def nSMK_model(u, e, n, t):
    V = get_SMK_V(n)
    e = dict(e)
    dim2id = dict(zip(V, range(len(V))))
    s = [None] * len(V)
    # Set variables from exo source
    for i in range(1,n+1):
        for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
            dim_id = dim2id[f"{dim}-U{i}"]
            s[dim_id] = e.get(f"{dim}-U{i}", u[dim_id])
    # Set depth-1
    for i in range(1,n+1):
        s[dim2id[f"GP-U{i}"]] = e.get(f"GP-U{i}", noise(s[dim2id[f"FS-U{i}"]] or s[dim2id[f"FN-U{i}"]], t))
        s[dim2id[f"GK-U{i}"]] = e.get(f"GK-U{i}", noise(s[dim2id[f"FF-U{i}"]] or s[dim2id[f"FDB-U{i}"]], t))
        s[dim2id[f"KMS-U{i}"]] = e.get(f"KMS-U{i}", noise(s[dim2id[f"A-U{i}"]] and s[dim2id[f"AD-U{i}"]], t))
    # Set DK
    for i in range(1,n+1):
        s[dim2id[f"DK-U{i}"]] = e.get(f"DK-U{i}", 
            noise(s[dim2id[f"GP-U{i}"]] and \
            s[dim2id[f"GK-U{i}"]] and \
            not any([s[dim2id[f"DK-U{j}"]] for j in range(1, i)]), t))
    # Set SD
    for i in range(1, n+1):
        s[dim2id[f"SD-U{i}"]] = e.get(f"SD-U{i}", 
            noise(s[dim2id[f"KMS-U{i}"]] and \
            not any([s[dim2id[f"SD-U{j}"]] for j in range(1,i)]), t))
    s[dim2id["DK"]] = e.get("DK", noise(any([s[dim2id[f"DK-U{i}"]] for i in range(1,n+1)]), t))
    s[dim2id["SD"]] = e.get("SD", noise(any([s[dim2id[f"SD-U{i}"]] for i in range(1,n+1)]), t))
    s[dim2id["SMK"]] = e.get("SMK", noise(s[dim2id["DK"]] or s[dim2id["SD"]], t))
    return s

def avg_nSMK_model(u, e, n, t, N):
    V = get_SMK_V(n)
    S = SMK_model_vectorized(u, [e], N, n, t)
    S = S.sum(axis=0) 
    return (S / N).tolist()

def get_avg_nSMK_SCM(n, u, N, nl):
    V = get_SMK_V(n)
    v = nSMK_model(u, [], n, t=0) # No noise for the instance
    t = nl/len(V) # Compute the noise threshold with the noise level

    def avg_vectorized_simulation(E):
        S = vectorized_SMK_model(u, E, n, N, t)
        out = []
        for i, e in enumerate(E):
            s = S[i*N:(i+1)*N].mean(axis=0).tolist()
            out.append((s, s[-1], psi(s)))
        return out
    
    
    return SCM(V=V,U=get_SMK_U(n),D=(0,1),F=lambda u,e: avg_nSMK_model(u, e, n, t, N),
               u=u,psi=psi,dag=get_SMK_DAG(n),v=v, sim=avg_vectorized_simulation)

def logistic(p, a=2):
    return 1 / (1 + np.exp(-a * (p - 0.5)))
    
def get_lucb_nSMK_SCM(n, u, nl, lucb_params):
    V = get_SMK_V(n)
    v = nSMK_model(u, [], n, t=0) # No noise for the instance
    t = nl/len(V) # Compute the noise threshold with the noise level

    def lucb_evaluator(E, N):
        S = vectorized_SMK_model(u, E, n, N, t)
        out = []
        for i, e in enumerate(E):
            s = S[i*N:(i+1)*N]
            phi_values = s[:,-1]
            psi_values = logistic((s.sum(axis=1)-1) / s.shape[1])
            out.append(np.stack([phi_values, psi_values]).T)
        return out
        
    return SCM(V=V,U=get_SMK_U(n),D=(0,1),F=lambda u,e: nSMK_model(u,e,n,t),u=u,psi=psi,
               dag=get_SMK_DAG(n),sim=lambda E: lucb(lucb_evaluator, E, **lucb_params),v=v,)