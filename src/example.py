from benchmark_models import *
from evaluation import *
from sanity import *


"""Suzzy example"""
print(f"\n\n\n\n{'*'*50}\n{'  Suzzy example  ':*^50}\n{'*'*50}\n")

scm_suzzy.find_causes(max_steps=-1,beam_size=3,early_stop=False,verbose=3)
scm_suzzy.show_identification_result()


"""SMK example"""
print(f"\n\n\n\n{'*'*50}\n{'  SMK example  ':*^50}\n{'*'*50}\n")

n = 3
u = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
scm = get_SMK_SCM(n, u)
print("  Run MBS, b=200, max_steps=6")
scm.find_causes(ISI=False, max_steps=6,beam_size=200, verbose=1)
scm.show_identification_result()

print("  Run ISI, b=200, max_steps=-1")
scm.find_causes(ISI=True, max_steps=-1,beam_size=200, verbose=1)
scm.show_identification_result()

print("  Expected:", smk_causes(scm.v))


"""Non Boolean SMK example"""
print(f"\n\n\n\n{'*'*50}\n{'  Non Boolean SMK example  ':*^50}\n{'*'*50}\n")

n = 3
u = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
scm_mSMK = get_mSMK_SCM(n, u)
print("  Run MBS, b=500, max_steps=7")
scm_mSMK.find_causes(ISI=False, beam_size=500, max_steps=7, verbose=1)
scm_mSMK.show_identification_result()

print("  Run MBS, b=100, max_steps=7")
scm_mSMK.find_causes(ISI=True, max_steps=7, beam_size=100, verbose=1)
scm_mSMK.show_identification_result()
print("  Expected:", smk_causes(scm.v))


"""Noisy SMK example"""
print(f"\n\n\n\n{'*'*50}\n{'  Noisy SMK example  ':*^50}\n{'*'*50}\n")
from experiments import lucb_params, nl
bs=64
lucb_params["beam_size"] = bs
nSuzzy_SCM = get_noisy_suzzy_SCM((1,1), .05, lucb_params)

print("  Run MBS + Naive sampling on Suzzy example")
nSuzzy_SCM.find_causes(epsilon=lucb_params["a"], verbose=2,
                       beam_size=lucb_params["beam_size"],max_steps=-1)
nSuzzy_SCM.show_identification_result()

u = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
n = 3
SCM_avg = get_avg_nSMK_SCM(n, u, lucb_params["max_iter"], nl)
np.random.seed(0)
print("  Run MBS + Naive sampling on SMK")
SCM_avg.find_causes(max_steps=7, beam_size=bs, verbose=1, epsilon=lucb_params["a"])
SCM_avg.show_identification_result()

print("  Run MBS + LUCB sampling on SMK")
SCM_lucb = get_lucb_nSMK_SCM(n, u, nl, lucb_params|{"verbose":1})
np.random.seed(0)
SCM_lucb.find_causes(max_steps=7, beam_size=bs, verbose=1, epsilon=lucb_params["a"])
SCM_lucb.show_identification_result()


"""Paper examples"""
print(f"\n\n\n\n{'*'*50}\n{'  Paper examples  ':*^50}\n{'*'*50}\n")

print("  Not model")
not_scm.find_causes(ISI=True, verbose=2)
not_scm.show_identification_result()

print("Expected:")
not_scm.find_causes(ISI=False, beam_size=-1, max_steps=-1)
not_scm.show_identification_result()
print()

print("  XOR model")
xor_scm.find_causes(ISI=True, verbose=2)
xor_scm.show_identification_result()

print("Expected:")
xor_scm.find_causes(ISI=False, beam_size=-1, max_steps=-1)
xor_scm.show_identification_result()
print()

print("  Chain model")
chain_scm.find_causes(ISI=True, verbose=2)
chain_scm.show_identification_result()
print("Expected:")
chain_scm.find_causes(ISI=False, beam_size=-1, max_steps=-1)
chain_scm.show_identification_result()
print()

print("  Split model")
split_scm.find_causes(ISI=True, verbose=2)
split_scm.show_identification_result()
print("Expected:")
split_scm.find_causes(ISI=False, beam_size=-1, max_steps=-1)
split_scm.show_identification_result()
print()

print("  OR model")
or_scm.find_causes(ISI=True, verbose=2)
or_scm.show_identification_result()
print("Expected:")
or_scm.find_causes(ISI=False, beam_size=-1, max_steps=-1)
or_scm.show_identification_result()
print()

"""Sanity checks"""
print(f"\n\n\n\n{'*'*50}\n{'  Sanity checks  ':*^50}\n{'*'*50}\n")
lines, index = run_sanity_checks(sanity_checks, verbose=False)
plot_sanity_checks(lines, index, show_pd = True)