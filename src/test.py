from benchmark_models import *
from evaluation import *
from sanity import sanity_checks


n = 3
# Example from the paper
# u = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
# Version with SD=1 (from 6 to 30 causes)
u = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0]

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
scm.find_causes(ISI=True, exhaustive=False, beam_size=252, max_steps=7, verbose=0)

scm.show_identification_result()

# 9915 / 676