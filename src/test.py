from benchmark_models import *
from evaluation import *
from sanity import sanity_checks
from experiments import nl, n_seeds


n = 3
# Example from the paper
u = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
# Version with SD=1 (from 6 to 30 causes)
# u = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0]

# # Random SMK initialization for more attakers
# np.random.seed(0)
# n=15
# u = np.random.randint(0, 2, size=6*n).tolist()

# Create the SCM
scm = get_SMK_SCM(n, u)
# print("Expected causes:", smk_causes(scm.v))

# # Print the SCM
# for variable, value in zip(scm_smk.V, scm_smk.v):
#     print(variable, value)

# Quick print
scm.find_causes(ISI=True, exhaustive=True, verbose=2, mbs_verbose=0)

scm.show_identification_result()

# data = load_json("results/noisy-full/structured-naive.json")
# datum = data[4]
# n, b = datum["n_attacker"], datum["beam_size"]
# print(f"{n=}, {b=}")
# # for res in datum["results"]:
# #     print(res["metrics"]["dice"])
# res = datum["results"][10]
# scm = get_avg_nSMK_SCM(n, res["context"], 50, nl)
# print(scm.u)
# for variable, value in zip(scm.V, scm.v):
#     print(variable, value)
# # for seed in range(n_seeds):
# np.random.seed(8)
# scm.find_causes(ISI=True, beam_size=b, max_steps=7, verbose=2, mbs_verbose=0, epsilon=.65)

# print()

# scm.show_identification_result(show_causes=False)