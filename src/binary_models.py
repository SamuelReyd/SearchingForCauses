from actualcauses import beam_search, show_rules
from benchmark_models import SMK_model, get_SMK_dim_labels, get_sympy_SMK

import numpy as np, timeit, time, matplotlib.pyplot as plt, json
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


### Utils
def show_all_vars(model, endo_vars, target):
    for var in endo_vars:
        if var.name == target.name:
            print(var.name, model.getVarByName(var.name).x)
        else:
            true_var = model.getVarByName(var.name)
            c1_var = model.getVarByName(f"C1_{var.name}")
            c2_var = model.getVarByName(f"C2_{var.name}")
            print(true_var.VarName, true_var.x, "C1:", c1_var.x, "C2:", c2_var.x)

def time_fn(fn, *args, **kargs):
    t = time.perf_counter()
    res = fn(*args, **kargs)
    return res, time.perf_counter() - t 

def format_scientific_latex(number):
    # Convert the number to scientific notation
    scientific_notation = f"{number:.1e}"

    # Split the scientific notation into the coefficient and exponent
    coefficient, exponent = scientific_notation.split('e')

    # Format the exponent in LaTeX style
    latex_exponent = f"e^{{{int(exponent)}}}"

    # Combine the coefficient and the LaTeX exponent
    latex_formatted = f"{coefficient}{latex_exponent}"

    return latex_formatted

### Beam Search setup
def simulate_boolean_model(rules, V_exo, apply_model, n_var, target=-1, **model_args):
    cf_values = []
    for rule in rules:
        interventions = dict(rule)
        s = np.zeros(n_var, dtype=int).tolist()
        apply_model(s, V_exo, interventions, **model_args)
        cf_values.append((s, s[target], sum(s)-1))
    return cf_values

def make_SCM(variables, V_exo, model, target=-1, **model_args):
    sim = lambda rules: simulate_boolean_model(
        rules, V_exo, model, len(variables), target, **model_args
    )
    instance = np.zeros(len(variables),dtype=bool).tolist()
    model(instance, V_exo, {}, **model_args)
    variables_domain = (0,1)
    domains = (variables_domain,)*(len(variables)-1)
    return {"instance": instance, "domains": domains, 
        "simulation": sim,  "variables": variables[:-1]}

def find_SCM(n_attacker, seed=None):
    if seed is not None: np.random.seed(seed)
    variables = get_SMK_dim_labels(n_attacker)
    SMK = False
    while not SMK:
        V_exo = np.random.randint(2,size=6 * n_attacker).astype(int)
        SCM = make_SCM(variables=variables, V_exo=V_exo,
            model=SMK_model,n_attacker=n_attacker)
        SMK = SCM["instance"][-1]
    return V_exo, SCM

def show_state(variables, state):
    print(" ".join(["~"*(1-value)+dim for dim, value in zip(variables, state)]))

def build_DAG(n_attackers, variables):
    dag = [[] for v in variables]
    leaf_vars = ["FS", "FN", "FF", "FDB", "A", "AD"]
    node_vars = ["GP", "GK", "KMS", "DK", "SD"]
    var2int = {v: i for i,v in enumerate(variables)}
    for n in range(1,n_attackers+1):
        dag[var2int[f'GP-U{n}']] += [var2int[f'FS-U{n}'], var2int[f'FN-U{n}']]
        dag[var2int[f'GK-U{n}']] += [var2int[f'FF-U{n}'], var2int[f'FDB-U{n}']]
        dag[var2int[f'KMS-U{n}']] += [var2int[f'A-U{n}'], var2int[f'AD-U{n}']]
        dag[var2int[f'DK-U{n}']] += [var2int[f'GP-U{n}'], var2int[f'GK-U{n}']] + [var2int[f'DK-U{j}'] for j in range(1, n)]
        dag[var2int[f'SD-U{n}']] += [var2int[f'KMS-U{n}']] + [var2int[f'SD-U{j}'] for j in range(1, n)]
        dag[var2int[f'DK']] += [var2int[f'DK-U{n}']]
        dag[var2int[f'SD']] += [var2int[f'SD-U{n}']]
        
    return dag, (len(dag)-1, len(dag)-2)

def build_DAG_non_boolean(n_attackers, variables):
    dag = [[] for v in variables]
    var2int = {v: i for i,v in enumerate(variables)}
    dag[var2int['GP']] += [var2int['FS'], var2int['FN']]
    dag[var2int['GK']] += [var2int['FF'], var2int['FDB']]
    dag[var2int['KMS']] += [var2int['A'], var2int['AD']]
    dag[var2int['DK']] += [var2int['GP'], var2int['GK']]
    dag[var2int['SD']] += [var2int['KMS']]
        
    return dag, (len(dag)-1, len(dag)-2)
