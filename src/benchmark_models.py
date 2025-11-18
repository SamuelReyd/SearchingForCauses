import math, numpy as np
from utils import generate_subsets, elementwise_any, elementwise_max
from sympy import symbols, Equivalent, to_cnf, Or, And, Not
# from actualcauses import lucb
from actualcauses_local.scm_class import SCM
from actualcauses_local.lucb import lucb
from actualcauses_local.simulator import simuator
from tqdm import tqdm
from collections import defaultdict

"""Base model"""


"""Rock Throwing"""
suzzy_vars = ["ST","BT","SH","BH","BS"]
suzzy_dag = {"ST":[],"BT":[],"SH":["ST"],"BH":["BT","SH"],"BS":["BH","SH"]}

class RockThrowingModel(BaseNumpyModel):
    def simulate(self, u):
        self["ST"] = u[0])
        self["BT"] = u[1])
        self["SH"] = s["ST"])
        self["BH"] = self["BT"] and not self["SH"]))
        self["BS"] = self["BH"] or self["SH"]))

scm_suzzy = SCM(V=suzzy_vars, U=["st","bt"], D=[(0,1)]*5, 
                model=RockThrowingModel(suzzy_vars), 
                u=(1,1), dag=suzzy_dag)

"""Examples from the ISI corectness proof"""
class OrModel(BaseNumpyModel):
    def simulate(self, u):
        self["X"] = u[0]
        self["A"] = self["X"]
        self["B"] = 1 and not self["A"]
        self["T"] = self["A"] or self["B"]
    
    
or_scm = SCM(V=["X", "A", "B", "T"],U=["x"],u=[1],D=(0,1),
             model=OrModel(["X", "A", "B", "T"]),
             dag={"X":[], "A":["X"], "B":["A"], "T":["A", "B"]})

class XNORModel(BaseNumpyModel):
    def simulate(self, u):
        self["X"] = u[0]
        self["A"] = self["X"]
        self["B"] = not self["X"]
        self["T"] = self["A"] != self["B"]
    
xnor_scm = SCM(V=["X", "A", "B", "T"],U=["x"],u=[1],D=(0,1),
               model=XNORModel(["X", "A", "B", "T"]),
               dag={"X":[], "A":["X"], "B":["X"], "T":["A", "B"]})

class ChainModel(BaseModel):
    def simulate(self, u):
        self["X"] = u[0]
        self["J"] = self["X"]
        self["H"] = self["J"]
        if self["H"] and not self["J"]: # H and not J
            self["G"] = "gx"
        elif (self["H"] and self["J"]) or (not self["H"] and self["J"]): # (H and J) or (not H and J)
            self["G"] = "g*"
        elif not self["H"] and not self["J"]: # not H and not J
            self["G"] = "g'"
        self["B"] = (s["G"] != "gx") or self["X"]
        self["T"] = self["B"]

chain_scm = SCM(V=["X", "J", "H", "G", "B", "T"],U=["x"],u=[1],
                D=[(0,1),(0,1),(0,1),("g'","g*", "gx"), (0,1), (0,1)],
                model=ChainModel(["X", "J", "H", "G", "B", "T"]),
                dag={"T":["B"], "B":["X","G"], "G": ["H", "J"], "H": ["J"], "J": ["X"], "X": []})

class SplitModel(BaseModel):
    def simulate(self, u):
        self["X"] = u[0]
        self["G"] = u[1]
        self["Y"] = u[2]
        self["H"] = u[3]
        if self["G"]: self["A"] = "g1"
        else: self["A"] = "x1" if self["X"] else "x0"
        if self["H"]: self["B"] = "h1"
        else: self["B"] = "y1" if self["Y"] else "y0"
        t = not (self["A"] == "x0" or self["B"] == "y0" or (self["A"]=="g1" and self["B"]=="h1"))
        self["T"] = t

split_scm = SCM(V=["X", "G", "Y", "H", "A", "B", "T"],U=["x", "g", "y", "h"],
                D=[(0,1), (0,1), (0,1), (0,1), ("x0","x1","g0","g1"),("y0","y1","h0","h1"),(0,1)],
                u=(1,0,1,0),
                model=SplitModel(["X", "G", "Y", "H", "A", "B", "T"]),
                dag={"X":[],"G":[],"Y":[],"H":[],"A":["X","G"],"B":["Y","H"],"T":["A","B"]})

"""Steal Master Key"""
smk_base_vars_exo = "fs fn ff fdb a ad".split()
smk_base_vars_users = "FS FN FF FDB A AD GP GK KMS DK SD".split()

def get_SMK_V(n):
    return [f"{dim}-U{i}" for dim in smk_base_vars_users for i in range(1,n+1)] + ["DK", "SD", "SMK"]

def get_SMK_U(n):
    return [f"{dim}-U{i}" for dim in smk_base_vars_exo for i in range(1,n+1)]

def SMKModel(BaseModel):
    def __init__(self, n, t, phi=None, psi=None):
        super().__init__(get_SMK_V(self.n)), phi, psi)
        self.n = n
        self.t = t

    def simulate(self, u):
        for i in range(1,self.n+1):
            for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
                self[f"{dim}-U{i}"] = u[self.dim2id[f"{dim}-U{i}"]]
        # Set depth-1
        for i in range(1,self.n+1):
            self[f"GP-U{i}"] = self[f"FS-U{i}"] | self[f"FN-U{i}"]
            self[f"GK-U{i}"] = self[f"FF-U{i}"] | self[f"FDB-U{i}"]
            self[f"KMS-U{i}"] = self[f"A-U{i}"] | self[f"AD-U{i}"]
        # Set depth-2
        for i in range(1,self.n+1):
            block = elementwise_any([self[f"DK-U{j}"] for j in range(1, i)])
            self[f"DK-U{i}"] = self[f"GP-U{i}"] & self[f"GK-U{i}") & ~block
        for i in range(1, self.n+1):
            block = elementwise_any([self[f"SD-U{j}"] for j in range(1,i)])
            self[f"SD-U{i}"] = self[f"KMS-U{i}"] & ~block
            
        self["DK"] = elementwise_any([self[f"DK-U{i}"] for i in range(1,self.n+1)])
        self["SD"] = elementwise_any([self[f"SD-U{i}"] for i in range(1,self.n+1)])
        self["SD"] = self["DK"] or self["SD"]
        
        
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

def get_SMK_SCM(n:int, u:list, heuristic=None):
    """
    Creates an SCM for the SMK scenario with n attackers and context u.
        n[int]: number of attackers
    """
    V = get_SMK_V(n)
    v = SMK_model(u, {}, n)
    if heuristic is not None:
        psi = lambda s: heuristic(s, v)
    else:
        psi = None
        
    return SCM(V=V,U=get_SMK_U(n),D=(0,1),
               u=u,dag=get_SMK_DAG(n), 
               model=SMKModel(V, psi=psi))

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
    # print(dict(zip(get_SMK_V(n),s)))
    # Set depth-1
    for i in range(1,n+1):
        s[dim2id[f"GP-U{i}"]] = e.get(f"GP-U{i}", s[dim2id[f"FS-U{i}"]] + s[dim2id[f"FN-U{i}"]])
        s[dim2id[f"GK-U{i}"]] = e.get(f"GK-U{i}", s[dim2id[f"FF-U{i}"]] + s[dim2id[f"FDB-U{i}"]])
        # Set KMS
        s[dim2id[f"KMS-U{i}"]] = e.get(f"KMS-U{i}", s[dim2id[f"A-U{i}"]] * s[dim2id[f"AD-U{i}"]])
    # print(dict(zip(get_SMK_V(n),s)))
    # Set depth-2
    for i in range(1,n+1):
        block = any([s[dim2id[f"DK-U{j}"]]>0 for j in range(1, i)]) if i > 1 else 0
        s[dim2id[f"DK-U{i}"]] = e.get(f"DK-U{i}", 
            (s[dim2id[f"GP-U{i}"]] * s[dim2id[f"GK-U{i}"]])>0 and not block)
    # print(dict(zip(get_SMK_V(n),s)))
    for i in range(1, n+1):
        block = any([s[dim2id[f"SD-U{j}"]]>0 for j in range(1, i)]) if i > 1 else 0
        s[dim2id[f"SD-U{i}"]] = e.get(f"SD-U{i}", s[dim2id[f"KMS-U{i}"]] and not block)
    # print(dict(zip(get_SMK_V(n),s)))
    # Set depth -3
    s[dim2id["DK"]] = e.get("DK", max([s[dim2id[f"DK-U{i}"]] for i in range(1,n+1)]))
    # print(dict(zip(get_SMK_V(n),s)))
    s[dim2id["SD"]] = e.get("SD", max([s[dim2id[f"SD-U{i}"]] for i in range(1,n+1)]))
    # print(dict(zip(get_SMK_V(n),s)))
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
        block_DK = elementwise_any([s[:,dim2id[f"DK-U{j}"]] > 0 for j in range(1, i)])
        set_value_vect(s, f"DK-U{i}", dim2id, 
                       (s[:,dim2id[f"GP-U{i}"]] * s[:,dim2id[f"GK-U{i}"]] > 0) | block_DK, 
                                  formated_E, t)
        block_SD = elementwise_any([s[:,dim2id[f"SD-U{j}"]] for j in range(1, i)])
        set_value_vect(s, f"SD-U{i}", dim2id, (s[:,dim2id[f"KMS-U{i}"]] > 0) | block_SD, formated_E)
    set_value_vect(s, "DK", dim2id, elementwise_any([s[:,dim2id[f"DK-U{j}"]] for j in range(1, n+1)]), formated_E, t)
    set_value_vect(s, "SD", dim2id, elementwise_any([s[:,dim2id[f"SD-U{j}"]] for j in range(1, n+1)]), formated_E, t)
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

"""Black Box SMK"""
def get_bbSMK_V(n):
    return [f"{label}-U{i}" for label in ("FS", "FN", "FF", "FDB", "A", "AD") for i in range(1,n+1)] + ["SMK"]

def bbSMK_model(u, e, n):
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
        s[dim2id[f"GP-U{i}"]] = int(s[dim2id[f"FS-U{i}"]] or s[dim2id[f"FN-U{i}"]])
        s[dim2id[f"GK-U{i}"]] = int(s[dim2id[f"FF-U{i}"]] or s[dim2id[f"FDB-U{i}"]])
        # Set KMS
        s[dim2id[f"KMS-U{i}"]] = int(s[dim2id[f"A-U{i}"]] and s[dim2id[f"AD-U{i}"]])
    # Set depth-2
    for i in range(1,n+1):
        s[dim2id[f"DK-U{i}"]] = int(s[dim2id[f"GP-U{i}"]] and s[dim2id[f"GK-U{i}"]] and not any([s[dim2id[f"DK-U{j}"]] for j in range(1, i)]))
    for i in range(1, n+1):
        s[dim2id[f"SD-U{i}"]] = int(s[dim2id[f"KMS-U{i}"]] and not any([s[dim2id[f"SD-U{j}"]] for j in range(1,i)]))
    # Set depth -3
    s[dim2id["DK"]] = int(any([s[dim2id[f"DK-U{i}"]] for i in range(1,n+1)]))
    s[dim2id["SD"]] = int(any([s[dim2id[f"SD-U{i}"]] for i in range(1,n+1)]))
    s[dim2id["SMK"]] = int(s[dim2id["DK"]] or s[dim2id["SD"]])
    return s[:len(get_bbSMK_V(n))] + s[-1:]

def vectorized_bbSMK_model(u, E, n, N=1, t=0):
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
            set_value_vect(s, f"{dim}-U{i}", dim2id, u[dim_id], formated_E)
    for i in range(1,n+1):
        s[:,dim2id[f"GP-U{i}"]] = s[:,dim2id[f"FS-U{i}"]] | s[:,dim2id[f"FN-U{i}"]]
        s[:,dim2id[f"GK-U{i}"]] = s[:,dim2id[f"FF-U{i}"]] | s[:,dim2id[f"FDB-U{i}"]]
        s[:,dim2id[f"KMS-U{i}"]] = s[:,dim2id[f"A-U{i}"]] & s[:,dim2id[f"AD-U{i}"]]
    for i in range(1,n+1):
        block_DK = elementwise_any([s[:,dim2id[f"DK-U{j}"]] for j in range(1, i)])
        s[:,dim2id[f"DK-U{i}"]] = s[:,dim2id[f"GP-U{i}"]] & s[:,dim2id[f"GK-U{i}"]] & ~block_DK
        block_SD = elementwise_any([s[:,dim2id[f"SD-U{j}"]] for j in range(1, i)])
        s[:,dim2id[f"SD-U{i}"]] = s[:,dim2id[f"KMS-U{i}"]] & ~block_SD
    s[:,dim2id["DK"]] = elementwise_any([s[:,dim2id[f"DK-U{j}"]] for j in range(1, n+1)])
    s[:,dim2id["SD"]] = elementwise_any([s[:,dim2id[f"SD-U{j}"]] for j in range(1, n+1)])
    s[:,dim2id["SMK"]] = s[:,dim2id["DK"]] | s[:,dim2id["SD"]]
    n_var = len(get_bbSMK_V(n))
    ret = np.ones((len(E)*N, n_var+1))
    ret[:, :n_var] = s[:,:n_var]
    ret[:,-1] = s[:,-1]
    return ret

def get_bbSMK_SCM(n, u):
    def vectorized_simulations(E):
        out = []
        sub_N = 100_000
        for i in range(0, len(E), sub_N):
            sub_E = E[i*sub_N:(i+1)*sub_N]
            S = vectorized_bbSMK_model(u, sub_E, n)
            for s in S:
                out.append((s, int(s[-1]), psi(s)))
        return out
    return SCM(V=get_bbSMK_V(n),U=get_SMK_U(n),D=[(0,1)] * (6 * n + 1),
               F=lambda u,e: bbSMK_model(u,e,n),u=u,psi=psi,dag=None,
              sim=vectorized_simulations)

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
               u=u,psi=psi,dag=get_SMK_DAG(n),v=v, 
               sim=avg_vectorized_simulation)

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
               dag=get_SMK_DAG(n),
               sim=lambda E: lucb(lucb_evaluator, E, **lucb_params),v=v,)