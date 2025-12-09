import numpy as np, time, matplotlib.pyplot as plt
from tqdm import tqdm

from .mbs import beam_search, get_rules, show_rules, get_sets

# This file is inspired by https://github.com/marcotcr/anchor

def kl_bernoulli(p, q):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))

def dup_bernoulli(mean, var, n, beta, max_iter=20, tol=1e-4):
    # print(f"{mean=}, {var=}, {n=}, {beta=}, {max_iter=}, {tol=}")
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

def dlow_bernoulli(mean, var, n, beta, max_iter=20, tol=1e-4):
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

def hoeffding_upper_bound(mean, var, n, beta):
    ub = mean + np.sqrt(beta / (2 * n))
    ub = np.minimum(1.0, ub)
    return ub

def hoeffding_lower_bound(mean, var, n, beta):
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

def plot_bound(stats, a, beam_size, beam_eps, cause_eps, non_cause_eps):
    phi_m, phi_ub, phi_lb, psi_m, psi_v, psi_M2, psi_ub, psi_lb, n = range(9)
    _, axes = plt.subplots(1,2,figsize=(8,4))
    # Plot phi
    sorted_ids = np.argsort(stats[phi_m])
    x = np.arange(len(sorted_ids))
    axes[1].plot(stats[phi_m][sorted_ids], label="mean")
    axes[1].plot(stats[phi_lb][sorted_ids], label="lower bound")
    axes[1].plot(stats[phi_ub][sorted_ids], label="upper bound")
    axes[1].fill_between(x, 
                     np.full(len(x), a) - cause_eps, 
                     np.full(len(x), a) + non_cause_eps, 
                     color='blue', alpha=0.2)
    axes[1].hlines(a, 0, stats.shape[1])
    axes[1].legend()

    # Plot psi
    sorted_ids = np.argsort(stats[psi_m])
    sorted_ids = [i for i in sorted_ids if stats[phi_m,i] >= a]
    x = np.arange(len(sorted_ids))
    axes[0].plot(stats[psi_m, sorted_ids])
    axes[0].plot(stats[psi_lb, sorted_ids])
    axes[0].plot(stats[psi_ub, sorted_ids])

    beam_ids = np.array(sorted_ids[:beam_size])
    non_beam_ids = np.array(sorted_ids[beam_size:])
    if not beam_ids.size or not non_beam_ids.size: return
    ut = beam_ids[np.argmax(stats[psi_ub,beam_ids])]
    lt = non_beam_ids[np.argmin(stats[psi_lb,non_beam_ids])]

    x_ut = np.where(np.array(sorted_ids) == ut)[0][0]
    x_lt = np.where(np.array(sorted_ids) == lt)[0][0]
    axes[0].scatter(x_ut, stats[psi_ub, ut], color='C2', zorder=5)
    axes[0].scatter(x_lt, stats[psi_lb, lt], color='C3', zorder=5)
        
    axes[0].fill_between(x, 
                         stats[psi_ub, ut] - beam_eps, 
                         stats[psi_lb, lt] + beam_eps, 
                         color='blue', alpha=0.2)
    axes[0].set_xscale('log')
    
    plt.show()

def update_bound(stats, fnt, bound_id, arm_ids, mean_id, variance_id, n_id, beta):
    var = stats[variance_id,arm_ids] if variance_id is not None else None
    stats[bound_id, arm_ids] = fnt(stats[mean_id,arm_ids], var, stats[n_id,arm_ids], beta)


def lucb(evaluator, rules, beam_size, a=.05, beam_eps=.1, cause_eps=.01, non_cause_eps=.01, 
         max_iter=200, verbose=0, batch_size=10, init_batch_size=20, lucb_info=None, delta=.1):

    init_batch_size = max(1, init_batch_size)
    n_arms = len(rules) # Doing armed bandits with the rules to evaluate

    n_stats = 9
    phi_m, phi_ub, phi_lb, psi_m, psi_v, psi_M2, psi_ub, psi_lb, n = range(n_stats)
    stats = np.zeros((n_stats,n_arms), dtype=float) 
    stats[phi_ub] = 1.0
    stats[psi_ub] = 1.0
    
    beta_phi = 0
    beta_psi = 0
    
    # Utils function
    def action_arms(arms, bs=batch_size):
        # Compute values
        E = [rules[arm] for arm in arms]
        if not len(E): return
        values_batchs = evaluator(E, bs)
        values_batchs = values_batchs.reshape(len(E), bs, -1)
        for arm, values_batch in zip(arms, values_batchs):
            # Update n 
            n_old = stats[n,arm]
            stats[n,arm] += bs
            # Compute batch stats
            mean_batch = np.mean(values_batch, axis=0)
            M2_batch = np.sum((values_batch[:,1] - mean_batch[1])**2) # Compute M2 only for psi
            # Update phi
            stats[phi_m,arm] += (mean_batch[0] - stats[phi_m, arm]) * bs / stats[n,arm]
            # Update psi
            stats[psi_m,arm] += (mean_batch[1] - stats[psi_m, arm]) * bs / stats[n,arm]
            delta = mean_batch[1] - stats[psi_m, arm]
            stats[psi_m,arm] += delta * bs / stats[n,arm]
            stats[psi_M2,arm] += M2_batch + delta**2 * n_old * bs / stats[n,arm]
            stats[psi_v,arm] = stats[psi_M2,arm] / stats[n,arm]
            
    def find_unsure_arms():
        cancel_ids = stats[phi_m] < a
        # Cause candidates
        cause_ids = np.argwhere(cancel_ids).flatten()
        if not cause_ids.size:
            cause_overlap = np.array([], dtype=int)
        else:
            ids_overlapping = stats[phi_ub, cause_ids] >= a + cause_eps
            cause_overlap = cause_ids[ids_overlapping]
        # Non-cause candidates
        non_cause_ids = np.argwhere(~cancel_ids).flatten()
        if not non_cause_ids.size:
            non_cause_overlap = np.array([], dtype=int)
        else:
            ids_overlapping = stats[phi_lb, non_cause_ids] <= a - non_cause_eps
            non_cause_overlap = non_cause_ids[ids_overlapping]
        # Beam overlap
        sorted_non_cause_ids = sorted(non_cause_ids, key = lambda i: stats[psi_m,i])
        beam_ids = np.array(sorted_non_cause_ids[:beam_size])
        non_beam_ids = np.array(sorted_non_cause_ids[beam_size:])
        if not non_beam_ids.size or not beam_ids.size:
            return cause_overlap, non_cause_overlap, np.array([], dtype=int), np.array([], dtype=int)
        ut = beam_ids[np.argmax(stats[psi_ub,beam_ids])]
        lt = non_beam_ids[np.argmin(stats[psi_lb,non_beam_ids])]
        # Beam candidates: overlap if the upper bound in the beam is higher than the lower bound outside the beam
        ids_overlapping = stats[psi_ub, beam_ids] >= stats[psi_lb, lt] + beam_eps  
        beam_overlap = beam_ids[ids_overlapping]
        # Non-beam candidates: overlap if the lower bound outside the beam is lower than the upper bound in the beam
        ids_overlapping = stats[psi_lb, non_beam_ids] <= stats[psi_ub, ut] - beam_eps
        nonbeam_overlap = non_beam_ids[ids_overlapping]
        return cause_overlap, non_cause_overlap, beam_overlap, nonbeam_overlap
    
    # Initialization
    beam_bound = 1
    cause_bound = 1
    non_cause_bound = 1
    
    it = 1
    # Loop
    with tqdm(total=n_arms * max_iter, disable=not verbose) as pbar:
        while stats[n].sum() < n_arms * max_iter:
            beta = compute_beta(n_arms, it, delta)
            # Check necessary arms
            unsure_arms = find_unsure_arms()
            cause_overlap, non_cause_overlap, beam_overlap, nonbeam_overlap = unsure_arms
            
            if verbose:
                print(f"cause: {len(cause_overlap)}, non cause: {len(non_cause_overlap)}, beam: {len(beam_overlap)}, non beam: {len(nonbeam_overlap)}, ")
            
            # Pull arms
            action_arms(
                np.unique(np.hstack(unsure_arms)), 
                init_batch_size if it == 1 else batch_size
            )
            # Update bounds
            # Compute the upper and lower bound for psi of everyone after the first batch to stabilize computations
            if True:#it == 1: 
                update_bound(stats, bernstein_upper_bound, psi_ub, np.arange(n_arms),   psi_m, psi_v, n, beta)
                update_bound(stats, bernstein_lower_bound, psi_lb, np.arange(n_arms),   psi_m, psi_v, n, beta)
            else:
                update_bound(stats, bernstein_upper_bound, psi_ub, beam_overlap,      psi_m, psi_v, n, beta)
                update_bound(stats, bernstein_lower_bound, psi_lb, nonbeam_overlap,   psi_m, psi_v, n, beta)
            nonbeam_overlap = np.arange(n_arms)
            update_bound(stats, dup_bernoulli,         phi_ub, cause_overlap,     phi_m, None,  n, beta)
            update_bound(stats, dlow_bernoulli,        phi_lb, non_cause_overlap, phi_m, None,  n, beta)
            
            # plot_bound(stats, a, beam_size, beam_eps, cause_eps, non_cause_eps)
            
            # Compute remaining overlap
            if not beam_overlap.size or not nonbeam_overlap.size: beam_bound = 0
            else: beam_bound = stats[psi_ub,beam_overlap].max() - stats[psi_lb,nonbeam_overlap].min()
                
            if not cause_overlap.size: cause_bound = 0
            else: cause_bound = stats[phi_ub,cause_overlap].max() - a
                
            if not non_cause_overlap.size: non_cause_bound = 0
            else: non_cause_bound = a - stats[phi_lb,non_cause_overlap].min()
            
            # Stop condition
            if beam_bound <= beam_eps and cause_bound <= cause_eps and non_cause_bound <= non_cause_eps: 
                if verbose > 1: 
                    print(f"Success: {beam_bound=:.4f} / {cause_bound=:.4f} / {non_cause_bound=:.4f})")
                break
            if cause_bound <= cause_eps and non_cause_bound <= non_cause_eps and (beam_size + (stats[phi_m] < a).sum()) >= n_arms:
                if verbose > 1:
                    print(f"All rules pass on to next state: {cause_bound=:.4f}, {non_cause_bound=:.4f}")
                break
            
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
    # print(f"Did {stats[n].sum()} calls (max={max_iter*len(rules)})")
    return stats[[phi_m,psi_m]].T
    # return [(stats[:,i], float(stats[phi_m,i]), float(stats[psi_m,i])) for i in range(n_arms)]
