import numpy as np, time
from tqdm import tqdm

from .base_algorithm import beam_search, get_rules, show_rules, get_sets

# This file is inspired by https://github.com/marcotcr/anchor

def kl_bernoulli(p, q):
    p = min(0.9999999999999999, max(0.0000001, p))
    q = min(0.9999999999999999, max(0.0000001, q))
    return (p * np.log(float(p) / q) + (1 - p) *
            np.log(float(1 - p) / (1 - q)))

def dup_bernoulli(p, level):
    lm = p
    um = min(min(1, p + np.sqrt(level / 2.)), 1)
    qm = (um + lm) / 2.
    if kl_bernoulli(p, qm) > level:
        um = qm
    else:
        lm = qm
    return um

def dlow_bernoulli(p, level):
    um = p
    lm = max(min(1, p - np.sqrt(level / 2.)), 0)
    qm = (um + lm) / 2.
    if kl_bernoulli(p, qm) > level:
        lm = qm
    else:
        um = qm
    return lm

def hoeffding_upper_bound(mean, n_samples, beta):
    """Upper confidence bound using Hoeffding's inequality for [0,1] bounded random variables."""
    if n_samples == 0:
        return 1.0
    return min(1.0, mean + np.sqrt(beta / (2 * n_samples)))

def hoeffding_lower_bound(mean, n_samples, beta):
    """Lower confidence bound using Hoeffding's inequality for [0,1] bounded random variables."""
    if n_samples == 0:
        return 0.0
    return max(0.0, mean - np.sqrt(beta / (2 * n_samples)))

def compute_beta(n_features, t):
    delta = .1
    alpha = 1.1
    k = 405.5
    temp = np.log(k * n_features * (t ** alpha) / delta)
    return temp + np.log(temp)



def lucb(evaluator, rules, beam_size, a=.05, beam_eps=.1, cause_eps=.01, non_cause_eps=.01, 
         max_iter=200, verbose=0, batch_size=10, init_batch_size=20, lucb_info=None):#, estim_phi_correct=0):
        
    n_arms = len(rules) # Doing armed bandits with the rules to evaluate

    s, m, ub, lb = range(4) # sum id, mean id, upper bound id, lower bound id
    phi = np.zeros((4,n_arms)) # sum, mean, ub, lb
    phi[ub] = 1
    psi = np.zeros((4,n_arms)) # sum, mean, ub, lb
    psi[ub] = 1
    # init_psi = np.zeros((init_batch_size,n_arms)) # init_runs
    n_samples = np.zeros(n_arms, dtype=int)
    
    beta = 0

    def make_initialization():
        if beam_size >= n_arms:
            ref_psi_value = 1
        else:
            ref_psi_value = sorted(init_psi.mean(0))[beam_size]
        psi[s] = (init_psi > ref_psi_value).sum(0)
        psi[m] = psi[s] / init_batch_size
        return ref_psi_value
    
    # Utils function
    def action_arm(arm, init=False):
        bs = init_batch_size if init else batch_size
        for i in range(bs):
            phi_value, psi_value = evaluator(rules[arm])
            phi[s,arm] += phi_value
            psi[s,arm] += psi_value #np.random.rand() < psi_value
            # if init:
            #     init_psi[i,arm] = psi_value
            # else:
            #     psi[s,arm] += (psi_value > ref_psi_value)
        n_samples[arm] += bs
        psi[m,arm] = psi[s,arm] / n_samples[arm]
        phi[m,arm] = phi[s,arm] / n_samples[arm]

    def update_bounds_beam(t):
        non_cause_ids = set(np.argwhere(phi[m] >= a).flatten())
        sorted_rule_ids = sorted(range(n_arms), key = lambda i: psi[m,i])
        sorted_non_cause_ids = np.array([i for i in sorted_rule_ids if i in non_cause_ids])
        
        beam_ids = sorted_non_cause_ids[:batch_size]
        non_beam_ids = sorted_non_cause_ids[batch_size:]
        if not beam_ids.size or not non_beam_ids.size: return 0
        for i in beam_ids:
            psi[ub,i] = hoeffding_upper_bound(psi[m,i], n_samples[i], beta)
        for i in non_beam_ids:
            psi[lb,i] = hoeffding_lower_bound(psi[m,i], n_samples[i], beta)
            
        ut = beam_ids[np.argmax(psi[ub,beam_ids])]
        lt = non_beam_ids[np.argmin(psi[lb,non_beam_ids])]
        B = psi[ub,ut] - psi[lb,lt]
        if B >= beam_eps:
            action_arm(ut)
            action_arm(lt)
        return B

    def update_bounds_non_cause(t):
        ids = np.argwhere(phi[m] >= a).flatten()
        # print(f"non cause: {ids=}")
        for i in ids:
            phi[lb,i] = dlow_bernoulli(phi[m,i], beta / n_samples[i])
        if not ids.size: return 0
        lt = ids[np.argmin(phi[lb,ids])]
        B = a - phi[lb,lt]
        if B >= non_cause_eps:
            action_arm(lt)
        return B

    def update_bounds_cause(t):
        ids = np.argwhere(phi[m] < a).flatten()
        # print(f"cause: {ids=}")
        for i in ids:
            phi[ub,i] = dup_bernoulli(phi[m,i], beta / n_samples[i])
        if not ids.size: return 0
        ut = ids[np.argmax(phi[ub,ids])]
        B = phi[ub,ut] - a
        if B >= cause_eps:
            action_arm(ut)
        return B
    
    # Initialization
    beam_bound = 1
    cause_bound = 1
    non_cause_bound = 1
    for arm in tqdm(range(n_arms), disable=not verbose):
        action_arm(arm, True)
    # Compute ref value as a threshold for making it into the beam or not
    # print(init_psi)
    # ref_psi_value = make_initialization()
    # if verbose == -1: print(f"ref psi {ref_psi_value:.2f}")
    # ref_psi_value += estim_phi_correct
    # if verbose == -1: print(f"ref psi corrected {ref_psi_value:.2f}")
    it = 1
    # Loop
    with tqdm(total=n_arms * max_iter, disable=not verbose) as pbar:
        while n_samples.sum() < n_arms * max_iter:
            # Stop condition
            if beam_bound <= beam_eps and cause_bound <= cause_eps and non_cause_bound <= non_cause_eps: 
                if verbose > 1: 
                    print(f"Success: {beam_bound=:.4f} / {cause_bound=:.4f} / {non_cause_bound=:.4f})")
                break
            if cause_bound <= cause_eps and non_cause_bound <= non_cause_eps and beam_size + (phi[m] < a).sum() >= n_arms:
                if verbose > 1:
                    print(f"All rules pass on to next state: {cause_bound=:.4f}, {non_cause_bound=:.4f}")
                break
                
            # Update bounds
            beta = compute_beta(n_arms, it)
            beam_bound = update_bounds_beam(it)
            cause_bound = update_bounds_cause(it)
            non_cause_bound = update_bounds_non_cause(it)
            pbar.n = n_samples.sum()
            pbar.refresh()
            it += 1
        else:
            # Render how much we fail if we fail to reach the bound
            if verbose > 1: 
                print(f"Fail: {beam_bound=:.4f} / {cause_bound=:.4f} / {non_cause_bound=:.4f}")
            
    if verbose > 2:
        print(f"ub={phi[ub].round(4)}")
        print(f"lb={phi[lb].round(4)}")
        print(f"preds={phi[m].round(2)}")
        print(f"n_samples={n_samples}")
    if lucb_info is not None:
        lucb_info.append({
            "n_calls": int(n_samples.sum())
        })
    outputs = [((n_sample, ub_i, lb_i, ub_s, lb_s), mean, mean_score) for \
                   n_sample, ub_i, lb_i, mean, ub_s, lb_s, mean_score in \
                    zip(
                        n_samples.tolist(),
                        phi[ub].tolist(), 
                        phi[lb].tolist(), 
                        phi[m].tolist(), 
                        psi[ub].tolist(), 
                        psi[lb].tolist(),
                        psi[m].tolist()
                    )]
    return outputs
