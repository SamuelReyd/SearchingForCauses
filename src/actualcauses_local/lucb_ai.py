import numpy as np
from tqdm import tqdm

def kl_bernoulli(p, q):
    p = min(0.9999999999999999, max(0.0000001, p))
    q = min(0.9999999999999999, max(0.0000001, q))
    return (p * np.log(float(p) / q) + (1 - p) *
            np.log(float(1 - p) / (1 - q)))

def dup_bernoulli(p, level):
    lm = p
    um = min(1, p + np.sqrt(level / 2.))
    qm = (um + lm) / 2.
    for _ in range(20):  # Add iteration limit
        if kl_bernoulli(p, qm) > level:
            um = qm
        else:
            lm = qm
        qm = (um + lm) / 2.
    return um

def dlow_bernoulli(p, level):
    um = p
    lm = max(0, p - np.sqrt(level / 2.))
    qm = (um + lm) / 2.
    for _ in range(20):  # Add iteration limit
        if kl_bernoulli(p, qm) > level:
            lm = qm
        else:
            um = qm
        qm = (um + lm) / 2.
    return lm

def compute_beta(n_features, t):
    delta = .1
    alpha = 1.1
    k = 405.5
    temp = np.log(k * n_features * (t ** alpha) / delta)
    return temp + np.log(temp)

def lucb(evaluator, rules, beam_size, a=.05, beam_eps=.1, cause_eps=.01, non_cause_eps=.01, 
         max_iter=200, verbose=0, batch_size=10, init_batch_size=50, lucb_info=None):
    """
    Fixed LUCB for continuous psi values.
    
    Key changes:
    - Increased init_batch_size default for better initial estimates
    - More conservative beam selection
    - Better handling of continuous psi values
    - Adaptive exploration strategy
    """
    
    n_arms = len(rules)
    
    s, m, ub, lb = range(4)
    phi = np.zeros((4, n_arms))
    phi[ub] = 1
    psi = np.zeros((4, n_arms))
    psi[ub] = 1
    n_samples = np.zeros(n_arms, dtype=int)
    
    # Track if arms are confidently classified
    cause_confident = np.zeros(n_arms, dtype=bool)
    non_cause_confident = np.zeros(n_arms, dtype=bool)
    
    def action_arm(arm, n_pulls=1):
        """Pull an arm n_pulls times and update statistics."""
        for i in range(n_pulls):
            phi_value, psi_value = evaluator(rules[arm])
            phi[s, arm] += phi_value
            psi[s, arm] += psi_value
        n_samples[arm] += n_pulls
        psi[m, arm] = psi[s, arm] / n_samples[arm]
        phi[m, arm] = phi[s, arm] / n_samples[arm]
    
    def update_confidence_bounds(t):
        """Update confidence bounds for all arms."""
        beta = compute_beta(n_arms, t)
        
        for i in range(n_arms):
            if n_samples[i] > 0:
                # Bounds for phi (cause/non-cause)
                phi[ub, i] = dup_bernoulli(phi[m, i], beta / n_samples[i])
                phi[lb, i] = dlow_bernoulli(phi[m, i], beta / n_samples[i])
                
                # Bounds for psi (beam selection)
                psi[ub, i] = dup_bernoulli(psi[m, i], beta / n_samples[i])
                psi[lb, i] = dlow_bernoulli(psi[m, i], beta / n_samples[i])
    
    def identify_confident_arms():
        """Identify arms that are confidently classified."""
        # Arms confidently below threshold a (causes)
        cause_confident[:] = phi[ub] < a
        
        # Arms confidently above threshold a (non-causes)
        non_cause_confident[:] = phi[lb] >= a
    
    def get_beam_candidates():
        """Get arms that could be in the beam (confidently non-causes)."""
        return np.where(non_cause_confident)[0]
    
    def select_arms_to_pull(t):
        """Select which arms to pull next based on uncertainty."""
        update_confidence_bounds(t)
        identify_confident_arms()
        
        arms_to_pull = []
        
        # 1. Pull uncertain cause/non-cause arms
        uncertain_phi = ~(cause_confident | non_cause_confident)
        if uncertain_phi.any():
            uncertain_ids = np.where(uncertain_phi)[0]
            # Pull the most uncertain (widest confidence interval)
            uncertainties = phi[ub, uncertain_ids] - phi[lb, uncertain_ids]
            most_uncertain = uncertain_ids[np.argmax(uncertainties)]
            arms_to_pull.append(('phi', most_uncertain))
        
        # 2. Among non-causes, identify beam membership
        beam_candidates = get_beam_candidates()
        
        if len(beam_candidates) > beam_size:
            # Need to identify top beam_size by psi
            # Pull arms with uncertain psi values
            psi_uncertainties = psi[ub, beam_candidates] - psi[lb, beam_candidates]
            
            # Focus on boundary arms (around the beam_size-th position)
            sorted_by_mean = beam_candidates[np.argsort(psi[m, beam_candidates])]
            
            # Consider arms around the boundary
            boundary_range = slice(max(0, beam_size - 2), 
                                  min(len(sorted_by_mean), beam_size + 2))
            boundary_arms = sorted_by_mean[boundary_range]
            
            # Pull the most uncertain boundary arm
            if len(boundary_arms) > 0:
                uncertainties = psi[ub, boundary_arms] - psi[lb, boundary_arms]
                most_uncertain = boundary_arms[np.argmax(uncertainties)]
                arms_to_pull.append(('psi', most_uncertain))
        
        return arms_to_pull
    
    def check_convergence():
        """Check if we've converged to confident answers."""
        update_confidence_bounds(it)
        identify_confident_arms()
        
        # All arms classified as cause or non-cause?
        all_classified = (cause_confident | non_cause_confident).all()
        if not all_classified:
            return False
        
        # Among non-causes, can we identify the beam?
        beam_candidates = get_beam_candidates()
        if len(beam_candidates) <= beam_size:
            return True
        
        # Check if beam membership is clear
        sorted_candidates = beam_candidates[np.argsort(psi[m, beam_candidates])]
        
        if len(sorted_candidates) > beam_size:
            # The beam_size-th arm's upper bound should be below 
            # the (beam_size+1)-th arm's lower bound
            beam_cutoff_ub = psi[ub, sorted_candidates[beam_size - 1]]
            non_beam_start_lb = psi[lb, sorted_candidates[beam_size]]
            
            return beam_cutoff_ub < non_beam_start_lb
        
        return True
    
    # Initialization - pull each arm init_batch_size times
    if verbose:
        print(f"Initialization: pulling each of {n_arms} arms {init_batch_size} times")
    
    for arm in tqdm(range(n_arms), disable=not verbose):
        action_arm(arm, init_batch_size)
    
    # Main loop
    it = 1
    with tqdm(total=n_arms * max_iter, disable=not verbose) as pbar:
        while n_samples.sum() < n_arms * max_iter:
            # Check convergence
            if check_convergence():
                if verbose > 1:
                    print(f"Converged at iteration {it}")
                break
            
            # Select and pull arms
            arms_to_pull = select_arms_to_pull(it)
            
            if not arms_to_pull:
                # If no specific arms selected, pull least-sampled arm
                min_samples = n_samples.min()
                candidates = np.where(n_samples == min_samples)[0]
                arm = np.random.choice(candidates)
                action_arm(arm, batch_size)
            else:
                for reason, arm in arms_to_pull:
                    action_arm(arm, batch_size)
            
            pbar.n = n_samples.sum()
            pbar.refresh()
            it += 1
        else:
            if verbose > 1:
                print(f"Reached max iterations")
    
    if verbose > 2:
        print(f"Final phi means: {phi[m].round(3)}")
        print(f"Final psi means: {psi[m].round(3)}")
        print(f"Samples per arm: {n_samples}")
    
    if lucb_info is not None:
        lucb_info.append({
            "n_calls": int(n_samples.sum()),
            "iterations": it
        })
    
    outputs = [((n_sample, ub_i, lb_i, ub_s, lb_s), mean, mean_score) 
               for n_sample, ub_i, lb_i, mean, ub_s, lb_s, mean_score in 
               zip(n_samples.tolist(), phi[ub].tolist(), phi[lb].tolist(), 
                   phi[m].tolist(), psi[ub].tolist(), psi[lb].tolist(), psi[m].tolist())]
    
    return outputs