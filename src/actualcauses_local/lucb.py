import numpy as np, time
from tqdm import tqdm

from .base_algorithm import beam_search, get_rules, show_rules, get_sets

# This file is inspired by https://github.com/marcotcr/anchor

def kl_bernoulli(p, q):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))

def dup_bernoulli(mean, n, beta, max_iter=20, tol=1e-4):
    level = beta / n
    valid_ub = (level > 0) & (mean < 1 - tol)
    ub = np.ones_like(mean)

    if np.any(valid_ub):
        lm_ub = mean[valid_ub].copy()
        um_ub = np.minimum(1.0, mean[valid_ub] + np.sqrt(level[valid_ub] / 2.))
    
        for _ in range(max_iter):
            qm = (um_ub + lm_ub) / 2.0
            kl = kl_bernoulli(mean[valid_ub], qm)
            above = kl > level[valid_ub]
            um_ub = np.where(above, qm, um_ub)
            lm_ub = np.where(above, lm_ub, qm)
            
            if np.all((um_ub - lm_ub) < tol):
                break
        ub[valid_ub] = um_ub
    return ub

def dlow_bernoulli(mean, n, beta, max_iter=20, tol=1e-4):
    level = beta / n
    valid_lb = (level > 0) & (mean > tol)
    lb = np.zeros_like(mean)
    
    if np.any(valid_lb):
        um_lb = mean[valid_lb].copy()
        lm_lb = np.maximum(0.0, mean[valid_lb] - np.sqrt(level[valid_lb] / 2.))
        
        for _ in range(max_iter):
            qm = (um_lb + lm_lb) / 2.0
            kl = kl_bernoulli(mean[valid_lb], qm)
            above = kl > level[valid_lb]
            lm_lb = np.where(above, qm, lm_lb)
            um_lb = np.where(above, um_lb, qm)
            
            if np.all((um_lb - lm_lb) < tol):
                break
        lb[valid_lb] = lm_lb
    return lb

def hoeffding_upper_bound(mean, n, beta):
    ub = mean + np.sqrt(beta / (2 * n))
    ub = np.minimum(1.0, ub)
    return ub

def hoeffding_lower_bound(mean, n, beta):
    lb = mean - np.sqrt(beta / (2 * n))
    lb = np.maximum(0.0, lb)
    return lb

def bernstein_lower_bound(mean, var, n, beta):
    eps = np.sqrt(2 * var * beta / n) + 2 * beta / (3 * n)
    return np.maximum(0.0, mean - eps)

def bernstein_upper_bound(mean, var, n, beta):
    eps = np.sqrt(2 * var * beta / n) + 2 * beta / (3 * n)
    return np.minimum(1.0, mean + eps)

def compute_beta_exact(n_arms, t, delta=0.1):
    return np.log(2 * n_arms * t**2 / delta)

def compute_beta_practical(n_arms, t, delta=0.1):
    return np.log(2 * n_arms / delta)

def compute_beta_usable(n_arms, t, delta=0.1):
    return np.log(2 / delta)

compute_beta = compute_beta_usable


def lucb(evaluator, rules, beam_size, a=.05, beam_eps=.1, cause_eps=.01, non_cause_eps=.01, 
         max_iter=200, verbose=0, batch_size=10, init_batch_size=20, lucb_info=None):#, estim_phi_correct=0):

    init_batch_size = max(1, init_batch_size)
    n_arms = len(rules) # Doing armed bandits with the rules to evaluate

    n_stats = 9
    # n_stats = 7
    phi_m, phi_ub, phi_lb, psi_m, psi_v, psi_M2, psi_ub, psi_lb, n = range(n_stats)
    # phi_m, phi_ub, phi_lb, psi_m, psi_ub, psi_lb, n = range(n_stats)
    stats = np.zeros((n_stats,n_arms), dtype=float) 
    stats[phi_ub] = 1.0
    stats[psi_ub] = 1.0
    
    beta_phi = 0
    beta_psi = 0
    
    # Utils function
    def action_arm(arm, bs=batch_size):
        # Compute values
        # values_batch = np.array([evaluator(rules[arm]) for _ in range(bs)]) # pairs (phi, psi)
        values_batch = evaluator(rules[arm], bs)
        # Update n 
        n_old = stats[n,arm]
        stats[n,arm] += bs
        # Compute batch stats
        mean_batch = np.mean(values_batch, axis=0)
        M2_batch = np.sum((values_batch[:,1] - mean_batch[1])**2) # Compute M2 only for psi
        # Update phi
        # print(f"{mean_batch[0]=:.2f}, {stats[phi_m, arm]=:.2f}, {bs / stats[n,arm]=:.2f}")
        stats[phi_m,arm] += (mean_batch[0] - stats[phi_m, arm]) * bs / stats[n,arm]
        # Update psi
        stats[psi_m,arm] += (mean_batch[1] - stats[psi_m, arm]) * bs / stats[n,arm]
        delta = mean_batch[1] - stats[psi_m, arm]
        stats[psi_m,arm] += delta * bs / stats[n,arm]
        stats[psi_M2,arm] += M2_batch + delta**2 * n_old * bs / stats[n,arm]
        stats[psi_v,arm] = stats[psi_M2,arm] / stats[n,arm]

    
    def update_bounds_beam(t):
        non_cause_ids = set(np.argwhere(stats[phi_m] >= a).flatten())
        sorted_rule_ids = sorted(range(n_arms), key = lambda i: stats[psi_m,i])
        sorted_non_cause_ids = np.array([i for i in sorted_rule_ids if i in non_cause_ids])
        
        beam_ids = sorted_non_cause_ids[:beam_size]
        non_beam_ids = sorted_non_cause_ids[beam_size:]
        if not beam_ids.size or not non_beam_ids.size: return 0

        # For the arms suspected to be in the beam: update upper bound
        stats[psi_ub,beam_ids] = bernstein_upper_bound(stats[psi_m,beam_ids], stats[psi_v,beam_ids], stats[n,beam_ids], beta_psi)
        # stats[psi_ub,beam_ids] = hoeffding_upper_bound(stats[psi_m,beam_ids], stats[n,beam_ids], beta_psi)
        
        # For the arms suspected not to be in the beam: update lower bound
        stats[psi_lb,non_beam_ids] = bernstein_lower_bound(stats[psi_m,non_beam_ids], stats[psi_v,non_beam_ids], stats[n,non_beam_ids], beta_psi)
        # stats[psi_lb,non_beam_ids] = hoeffding_lower_bound(stats[psi_m,non_beam_ids], stats[n,non_beam_ids], beta_psi)

        # Compute the id of the higher upper bound of the candidate beams
        #  + the id of the smallest lower bound of the candidate non-beam
        ut = beam_ids[np.argmax(stats[psi_ub,beam_ids])]
        lt = non_beam_ids[np.argmin(stats[psi_lb,non_beam_ids])]

        beam_overlap = stats[psi_ub, beam_ids] - (stats[psi_lb, lt] - beam_bound)
        nonbeam_overlap = (stats[psi_ub, ut] + beam_bound) - stats[psi_lb, non_beam_ids]

        beam_to_pull = beam_ids[beam_overlap >= 0]
        nonbeam_to_pull = non_beam_ids[nonbeam_overlap >= 0]

        n_select = -1
        beam_to_pull = beam_to_pull[np.argsort(-beam_overlap[beam_overlap >= 0])[:n_select]]
        nonbeam_to_pull = nonbeam_to_pull[np.argsort(-nonbeam_overlap[nonbeam_overlap >= 0])[:n_select]]

        # Merge arms to pull
        pull_ids = np.unique(np.concatenate([beam_to_pull, nonbeam_to_pull]))
        for i in pull_ids:
            action_arm(i)
        
        return stats[psi_ub,ut] - stats[psi_lb,lt]

    def update_bounds_non_cause(t):
        ids = np.argwhere(stats[phi_m] >= a).flatten()
        if not ids.size: return 0
        stats[phi_lb,ids] = dlow_bernoulli(stats[phi_m,ids], stats[n,ids], beta_phi)
        lt = ids[np.argmin(stats[phi_lb,ids])]
        pull_ids = ids[stats[phi_lb, ids] <= a + non_cause_bound]
        for i in pull_ids:
            action_arm(i)
        return a - stats[phi_lb,lt]

    def update_bounds_cause(t):
        ids = np.argwhere(stats[phi_m] < a).flatten()
        if not ids.size: return 0
        stats[phi_ub,ids] = dup_bernoulli(stats[phi_m,ids], stats[n,ids], beta_phi)
        ut = ids[np.argmax(stats[phi_ub,ids])]
        pull_ids = ids[stats[phi_ub, ids] >= a - cause_bound]
        for i in pull_ids:
            action_arm(i)
        return stats[phi_ub,ut] - a
    
    # Initialization
    beam_bound = 1
    cause_bound = 1
    non_cause_bound = 1

    for arm in tqdm(range(n_arms), disable=not verbose):
        action_arm(arm, init_batch_size)
    it = 1
    # Loop
    with tqdm(total=n_arms * max_iter, disable=not verbose) as pbar:
        while stats[n].sum() < n_arms * max_iter:
            # Stop condition
            if beam_bound <= beam_eps and cause_bound <= cause_eps and non_cause_bound <= non_cause_eps: 
                if verbose > 1: 
                    print(f"Success: {beam_bound=:.4f} / {cause_bound=:.4f} / {non_cause_bound=:.4f})")
                break
            if cause_bound <= cause_eps and non_cause_bound <= non_cause_eps and (beam_size + (stats[phi_m] < a).sum()) >= n_arms:
                if verbose > 1:
                    print(f"All rules pass on to next state: {cause_bound=:.4f}, {non_cause_bound=:.4f}")
                break
                
            # Update bounds
            # Values for delta should be fine-tuned
            beta_psi = compute_beta(n_arms, it, delta=.1)
            beta_phi = compute_beta(n_arms, it, delta=.1)
            
            beam_bound = update_bounds_beam(it)
            cause_bound = update_bounds_cause(it)
            non_cause_bound = update_bounds_non_cause(it)
            
            pbar.n = stats[n].sum()
            pbar.refresh()
            it += 1
        else:
            # Render how much we fail if we fail to reach the bound
            if verbose > 1: 
                print(f"Fail: {beam_bound=:.4f} / {cause_bound=:.4f} / {non_cause_bound=:.4f}")
            
    if verbose > 2:
        print(f"phi ub={stats[phi_ub].round(2)}")
        print(f"phi lb={stats[phi_lb].round(2)}")
        print(f"phi m={stats[phi_m].round(2)}")
        print(f"psi ub={stats[psi_ub].round(2)}")
        print(f"psi lb={stats[psi_lb].round(2)}")
        print(f"psi v={stats[psi_v].round(2)}")
        print(f"psi m={stats[psi_m].round(2)}")
        print(f"n_samples={stats[n]}")
    if lucb_info is not None:
        lucb_info.append({
            "n_calls": int(n_samples.sum())
        })
    return [(stats[:,i], float(stats[phi_m,i]), float(stats[psi_m,i])) for i in range(n_arms)]
