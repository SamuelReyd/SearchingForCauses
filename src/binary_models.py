# from actualcauses import beam_search, show_rules
# from benchmark_models import SMK_model, get_SMK_V, get_sympy_SMK

# import numpy as np, timeit, time, matplotlib.pyplot as plt, json
# from tqdm import tqdm
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from scipy.optimize import curve_fit


# from general import *


# ### Utils
# def show_all_vars(model, endo_vars, target):
#     for var in endo_vars:
#         if var.name == target.name:
#             print(var.name, model.getVarByName(var.name).x)
#         else:
#             true_var = model.getVarByName(var.name)
#             c1_var = model.getVarByName(f"C1_{var.name}")
#             c2_var = model.getVarByName(f"C2_{var.name}")
#             print(true_var.VarName, true_var.x, "C1:", c1_var.x, "C2:", c2_var.x)

# def time_fn(fn, *args, **kargs):
#     t = time.perf_counter()
#     res = fn(*args, **kargs)
#     return res, time.perf_counter() - t 

# def format_scientific_latex(number):
#     # Convert the number to scientific notation
#     scientific_notation = f"{number:.1e}"

#     # Split the scientific notation into the coefficient and exponent
#     coefficient, exponent = scientific_notation.split('e')

#     # Format the exponent in LaTeX style
#     latex_exponent = f"e^{{{int(exponent)}}}"

#     # Combine the coefficient and the LaTeX exponent
#     latex_formatted = f"{coefficient}{latex_exponent}"

#     return latex_formatted

# def find_SCM(n_attacker, seed=None):
#     if seed is not None: np.random.seed(seed)
#     variables = get_SMK_V(n_attacker)
#     SMK = False
#     while not SMK:
#         V_exo = np.random.randint(2,size=6 * n_attacker).astype(int)
#         SCM = make_SCM(variables=variables, V_exo=V_exo,
#             model=SMK_model,n_attacker=n_attacker)
#         SMK = SCM["instance"][-1]
#     return V_exo, SCM

# def show_state(variables, state):
#     print(" ".join(["~"*(1-value)+dim for dim, value in zip(variables, state)]))
