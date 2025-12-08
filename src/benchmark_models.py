import math, numpy as np
from utils import generate_subsets, elementwise_any, elementwise_max
from sympy import symbols, Equivalent, to_cnf, Or, And, Not
from actualcauses_local.scm import SCM
from actualcauses_local.lucb import lucb
from actualcauses_local.system_model import *
from tqdm import tqdm
from collections import defaultdict

"""Base model"""


"""Rock Throwing"""
suzzy_vars = ["ST","BT","SH","BH","BS"]
suzzy_dag = {"ST":[],"BT":[],"SH":["ST"],"BH":["BT","SH"],"BS":["BH","SH"]}

class RockThrowingModel(BaseNumpyModel):
    def simulate(self, u):
        self["ST"] = u[0]
        self["BT"] = u[1]
        self["SH"] = self["ST"]
        self["BH"] = self["BT"] & ~self["SH"]
        self["BS"] = self["BH"] | self["SH"]

scm_suzzy = SCM(V=suzzy_vars, U=["st","bt"], D=[(0,1)]*5, 
                model=RockThrowingModel(suzzy_vars), 
                u=(1,1), dag=suzzy_dag)

"""Examples from the ISI corectness proof"""
class OrModel(BaseNumpyModel):
    def simulate(self, u):
        self["X"] = u[0]
        self["A"] = self["X"]
        self["B"] = 1 & ~self["A"]
        self["T"] = self["A"] | self["B"]
    
    
or_scm = SCM(V=["X", "A", "B", "T"],U=["x"],u=[1],D=(0,1),
             model=OrModel(["X", "A", "B", "T"]),
             dag={"X":[], "A":["X"], "B":["A"], "T":["A", "B"]})

class XORModel(BaseNumpyModel):
    def simulate(self, u):
        self["X"] = u[0]
        self["A"] = self["X"]
        self["B"] = ~self["X"]
        self["T"] = self["A"] != self["B"]
    
xor_scm = SCM(V=["X", "A", "B", "T"],U=["x"],u=[1],D=(0,1),
               model=XORModel(["X", "A", "B", "T"]),
               dag={"X":[], "A":["X"], "B":["X"], "T":["A", "B"]})

chain_vars = ["X", "J", "H", "G", "B", "T"]

class ChainModel(BaseNumpyModel):
    def __init__(self):
        psi = lambda s: np.ones(s.shape[0])
        super().__init__(chain_vars, phi=None, psi=psi, dtype=object)
        
    def simulate(self, u):
        self["X"] = u[0]
        self["J"] = self["X"]
        self["H"] = self["J"]
        g = np.zeros((self.S.shape[0]), dtype=object)
        g_x_ids = self["H"].astype(bool) & ~self["J"].astype(bool)
        g[g_x_ids] = "gx"
        g_star_ids = (self["H"].astype(bool) & self["J"].astype(bool)) | (~self["H"].astype(bool) & self["J"].astype(bool))
        g[g_star_ids] = "g*"
        g_prime_ids = ~self["H"].astype(bool) & ~self["J"].astype(bool)
        g[g_prime_ids] = "g'"
        self["G"] = g
        self["B"] = (self["G"] != "gx") | self["X"].astype(bool)
        self["T"] = self["B"]

chain_scm = SCM(V=chain_vars,U=["x"],u=[1],model=ChainModel(),
                D=[(0,1),(0,1),(0,1),("g'","g*", "gx"), (0,1), (0,1)],
                dag={"T":["B"], "B":["X","G"], "G": ["H", "J"], "H": ["J"], "J": ["X"], "X": []})

split_vars = ["X", "G", "Y", "H", "A", "B", "T"]

class SplitModel(BaseNumpyModel):
    def __init__(self):
        psi = lambda s: np.ones(s.shape[0])
        super().__init__(split_vars, phi=None, psi=psi, dtype=object)
        
    def simulate(self, u):
        self["X"] = u[0]
        self["G"] = u[1]
        self["Y"] = u[2]
        self["H"] = u[3]
        
        a = np.zeros((self.S.shape[0]), dtype=object)
        g1_ids = self["G"].astype(bool)
        a[g1_ids] = "g1"
        x1_ids = ~g1_ids & self["X"].astype(bool)
        a[x1_ids] = "x1"
        x0_ids = ~g1_ids & ~self["X"].astype(bool)
        a[x0_ids] = "x0"
        self["A"] = a
        
        b = np.zeros((self.S.shape[0]), dtype=object)
        h1_ids = self["H"].astype(bool)
        b[g1_ids] = "h1"
        y1_ids = ~h1_ids & self["Y"].astype(bool)
        b[y1_ids] = "y1"
        y0_ids = ~h1_ids & ~self["Y"].astype(bool)
        b[y0_ids] = "y0"
        self["B"] = b
        
        t = ~(
            (self["A"] == "x0") | 
            (self["B"] == "y0") | 
            ((self["A"]=="g1") & (self["B"]=="h1"))
        )
        self["T"] = t

split_scm = SCM(V=split_vars,U=["x", "g", "y", "h"],u=(1,0,1,0), model=SplitModel(),
                D=[(0,1), (0,1), (0,1), (0,1), ("x0","x1","g0","g1"),("y0","y1","h0","h1"),(0,1)],
                dag={"X":[],"G":[],"Y":[],"H":[],"A":["X","G"],"B":["Y","H"],"T":["A","B"]})

"""Steal Master Key"""
smk_base_vars_exo = "fs fn ff fdb a ad".split()
smk_base_vars_users = "FS FN FF FDB A AD GP GK KMS DK SD".split()

def get_SMK_V(n):
    return [f"{dim}-U{i}" for dim in smk_base_vars_users for i in range(1,n+1)] + ["DK", "SD", "SMK"]

def get_SMK_U(n):
    return [f"{dim}-U{i}" for dim in smk_base_vars_exo for i in range(1,n+1)]

class SMKModel(BaseNumpyModel):
    def __init__(self, n, phi=None, psi=None, dtype=None):
        super().__init__(get_SMK_V(n), phi=phi, psi=psi, dtype=dtype)
        self.n = n

    def simulate(self, u):
        for i in range(1,self.n+1):
            for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
                self[f"{dim}-U{i}"] = u[self.dim2id[f"{dim}-U{i}"]]
        for i in range(1,self.n+1):
            self[f"GP-U{i}"] = self[f"FS-U{i}"] | self[f"FN-U{i}"]
            self[f"GK-U{i}"] = self[f"FF-U{i}"] | self[f"FDB-U{i}"]
            self[f"KMS-U{i}"] = self[f"A-U{i}"] & self[f"AD-U{i}"]
        for i in range(1,self.n+1):
            block = elementwise_any([self[f"DK-U{j}"] for j in range(1, i)])
            self[f"DK-U{i}"] = self[f"GP-U{i}"] & self[f"GK-U{i}"] & ~block
        for i in range(1, self.n+1):
            block = elementwise_any([self[f"SD-U{j}"] for j in range(1,i)])
            self[f"SD-U{i}"] = self[f"KMS-U{i}"] & ~block
            
        self["DK"] = elementwise_any([self[f"DK-U{i}"] for i in range(1,self.n+1)])
        self["SD"] = elementwise_any([self[f"SD-U{i}"] for i in range(1,self.n+1)])
        self["SMK"] = self["DK"] | self["SD"]
        
"""Base and noisy SMK"""
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

def get_SMK_SCM(n:int, u:list, t: float=0, heuristic=None):
    """
    Creates an SCM for the SMK scenario with n attackers and context u.
        n[int]: number of attackers
    """
    V = get_SMK_V(n)
    model=SMKModel(n)
    v = model(u, {})
    if heuristic is not None:
        psi = lambda s: heuristic(s, v)
    else:
        psi = None
        
    return SCM(V=V,U=get_SMK_U(n),D=(0,1),
               u=u,dag=get_SMK_DAG(n), 
               model=SMKModel(n, psi=psi, phi=None))

"""Non-boolean SMK"""
mSMK_variables = smk_base_vars_users + ["SMK"]

class mSMKModel(SMKModel):
    def __init__(self, n, phi=None, psi=None):
        super().__init__(n, phi=phi, psi=psi, dtype=int)
    
    def simulate(self, u):
        for i in range(1,self.n+1):
            for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
                self[f"{dim}-U{i}"] = u[self.dim2id[f"{dim}-U{i}"]]
        for i in range(1,self.n+1):
            self[f"GP-U{i}"] = self[f"FS-U{i}"] + self[f"FN-U{i}"]
            self[f"GK-U{i}"] = self[f"FF-U{i}"] + self[f"FDB-U{i}"]
            self[f"KMS-U{i}"] = self[f"A-U{i}"] * self[f"AD-U{i}"]
        for i in range(1,self.n+1):
            block = elementwise_any([self[f"DK-U{j}"]>0 for j in range(1, i)]) if i > 1 else np.False_
            self[f"DK-U{i}"] = ((self[f"GP-U{i}"] * self[f"GK-U{i}"])>0) & ~block
        for i in range(1, self.n+1):
            block = elementwise_any([self[f"SD-U{j}"]>0 for j in range(1, i)]) if i > 1 else np.False_
            self[f"SD-U{i}"] = self[f"KMS-U{i}"] & ~block
        self["DK"] = elementwise_any([self[f"DK-U{i}"] for i in range(1,self.n+1)])
        self["SD"] = elementwise_any([self[f"SD-U{i}"] for i in range(1,self.n+1)])
        self["SMK"] = (self["DK"]>0) | (self["SD"]>0)

def get_mSMK_domains(n):
    # FS, FN, FF, FDB, A, AD / GP, GK / KMS / DKu / SDu / DK / SD, SMK
    return 6 * n * [(0,1)] + 2 * n * [(0,1,2)] + n * [(0,1)] + n * [(0,1,2,3,4)] + n * [(0,1)] + [(0,1,2,3,4)] + 2 * [(0,1)]

def get_mSMK_SCM(n, u):
    V = get_SMK_V(n)
        
    return SCM(V=get_SMK_V(n),U=get_SMK_U(n),
               D=get_mSMK_domains(n),u=u,dag=get_SMK_DAG(n),
               model=mSMKModel(n))

"""Black Box SMK"""
def get_bbSMK_V(n):
    return [f"{label}-U{i}" for label in ("FS", "FN", "FF", "FDB", "A", "AD") for i in range(1,n+1)] + ["SMK"]

class bbSMKModel(BaseNumpyModel):
    def __init__(self, n):
        super().__init__(get_bbSMK_V(n))
        self.n = n

    def simulate(self, u):
        for i in range(1,self.n+1):
            for dim in ("FS", "FN", "FF", "FDB", "A", "AD"):
                self[f"{dim}-U{i}"] = u[self.dim2id[f"{dim}-U{i}"]]
        V = get_SMK_V(self.n)
        s = np.zeros((self.S.shape[0], len(V)), dtype=bool)
        dim2id = dict(zip(V, range(len(V))))
        for i in range(1,self.n+1):
            s[:,dim2id[f"GP-U{i}"]] = self[f"FS-U{i}"] | self[f"FN-U{i}"]
            s[:,dim2id[f"GK-U{i}"]] = self[f"FF-U{i}"] | self[f"FDB-U{i}"]
            s[:,dim2id[f"KMS-U{i}"]] = self[f"A-U{i}"] & self[f"AD-U{i}"]
        for i in range(1,self.n+1):
            block_DK = elementwise_any([s[:,dim2id[f"DK-U{j}"]] for j in range(1, i)])
            s[:,dim2id[f"DK-U{i}"]] = s[:,dim2id[f"GP-U{i}"]] & s[:,dim2id[f"GK-U{i}"]] & ~block_DK
            block_SD = elementwise_any([s[:,dim2id[f"SD-U{j}"]] for j in range(1, i)])
            s[:,dim2id[f"SD-U{i}"]] = s[:,dim2id[f"KMS-U{i}"]] & ~block_SD
        s[:,dim2id["DK"]] = elementwise_any([s[:,dim2id[f"DK-U{j}"]] for j in range(1, self.n+1)])
        s[:,dim2id["SD"]] = elementwise_any([s[:,dim2id[f"SD-U{j}"]] for j in range(1, self.n+1)])
        self["SMK"] = s[:,dim2id["DK"]] | s[:,dim2id["SD"]]


def get_bbSMK_SCM(n, u):
    return SCM(V=get_bbSMK_V(n),U=get_SMK_U(n),D=[(0,1)] * (6 * n + 1),u=u,dag=None,model=SMKModel(n))

"""Noisy SMK"""
class AvgRockThrowingModel(AverageNumpyModel):
    def simulate(self, u): 
        return RockThrowingModel.simulate(self, u)
        
class LUCBRockThrowingModel(LUCBNumpyModel):
    def simulate(self, u): 
        return RockThrowingModel.simulate(self, u)


def get_noisy_suzzy_SCM(u, t, lucb_params):
    return SCM(V=suzzy_vars,U=("st", "bt"),D=[(0,1)] * len(suzzy_vars),
               u=(1,1),dag=None,
               model=LUCBRockThrowingModel(suzzy_vars, t, lucb_params))

class AvgSMKModel(AverageNumpyModel, SMKModel):
    def __init__(self, n, t, N, phi=None, psi=None):
        SMKModel.__init__(self, n, phi=phi, psi=psi)
        V = get_SMK_V(n)
        AverageNumpyModel.__init__(self, V, t, N, phi, psi)

class LUCBSMKModel(LUCBNumpyModel, SMKModel):
    def __init__(self, n, t, lucb_params, phi=None, psi=None):
        SMKModel.__init__(self, n, phi=phi, psi=psi)
        V = get_SMK_V(n)
        LUCBNumpyModel.__init__(self, V, t, lucb_params, phi=phi, psi=psi)

def get_avg_nSMK_SCM(n, u, N, nl):
    V = get_SMK_V(n)
    v = SMKModel(n)(u, []) # No noise for the instance
    t = nl/len(V) # Compute the noise threshold with the noise level

    return SCM(V=V,U=get_SMK_U(n),D=(0,1),u=u,dag=get_SMK_DAG(n),v=v,
               model=AvgSMKModel(n, t, N))
    
def get_lucb_nSMK_SCM(n, u, nl, lucb_params):
    V = get_SMK_V(n)
    v = SMKModel(n)(u, []) # No noise for the instance
    t = nl/len(V) # Compute the noise threshold with the noise level
    return SCM(V=V,U=get_SMK_U(n),D=(0,1),u=u,dag=get_SMK_DAG(n),v=v,
              model=LUCBSMKModel(n, t, lucb_params))