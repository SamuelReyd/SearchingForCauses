import numpy as np, bisect, time
from tqdm import tqdm
from itertools import count
from collections import defaultdict

def render_step(verbose, causes, non_causes, instance, variables):
    if len(non_causes):
        if verbose == 2:
            print("Number of causes found:", len(causes))
            print("Number of non-causes remaining:", len(non_causes))
            print("Best non-cause:")
            show_rule(non_causes[0], variables)
            print("Worst non-cause:")
            show_rule(non_causes[-1], variables)
        if verbose >= 3:
            print("Causes found this step:")
            show_rules(causes, variables)
            print("Rules passed for next step:")
            show_rules(non_causes, variables)
    else:
        if verbose >= 2:
            print("No rule available")

def render_time(time):
    n_hours = time//(60*60)
    ret = ""
    if int(n_hours): ret += f"{n_hours:.0f}h"
    if int(n_hours) or int((time%3600)//60): ret += f"{(time%3600)//60:.0f}min"
    if not int(n_hours): ret += f"{time%60:.0f}s"
    if not (int(n_hours) or int((time%3600)//60)): ret += f"{(time-int(time))*100:.0f}ms"
    return ret

def show_raw_rule(rule):
    C, W = get_sets(rule, instance)
    s, p = 0,0
    show_rule((rule, s, p, C, W, None), variables)

def get_sets(rule, instance):
    C = set()
    W = set()
    for feature, value in rule:
        if instance[feature] != value:
            C.add(feature)
        else:
            W.add(feature)
    return C, W

def sort_key(rule_values):
    _, rule_output, rule_score, C, W, _ = rule_values
    return (rule_score, len(C), C, len(W), W)

def format_value(value):
    if isinstance(value, float):
        return f"{float(value):.2f}"
    elif isinstance(value, int):
        return f"{int(value)}"
    else:
        return f"{value}"

def get_rule_desc(rule_values, variables, show_score=False):
    rule, output, score, C, W, _ = rule_values
    dim2value = dict(rule)
    C = {variables[c]:format_value(dim2value[c]) for c in C}
    W = {variables[w]:format_value(dim2value[w]) for w in W}
    if show_score: 
        if isinstance(score, tuple):
            return f"{C=}, {W=}, {output=}, {score=}"
        return f"{C=}, {W=}, {output=}, {score=:.3f}"
    return f"{C=}, {W=}"

def show_rule(rule_values, variables):
    print(get_rule_desc(rule_values, variables, True))
    
def show_rules(rule_values, variables):
    for r_values in rule_values:
        show_rule(r_values, variables)

def get_initial_rules(instance, domains):
    rules = []
    for feature, domain in enumerate(domains):
        for value in domain:
            if instance[feature] != value:
                rules.append(((feature, value),))
    return rules

def filter_minimality(candidates):
    # Remove causes with strict subsets that are causes
    Cs = [values[3] for values in candidates]
    candidates = [values for values in candidates if not any([c < values[3] for c in Cs])]

    # If there are equalities, keep the ones with smallest W and best score
    Cs = defaultdict(lambda: [])
    for rule_values in candidates:
        Cs[tuple(rule_values[3])].append(rule_values)
    causes = []
    for cands in Cs.values():
        best = min(cands, key=lambda rule_values: (len(rule_values[4]),rule_values[2]))
        causes.append(best)
    return causes
    

def get_rules(previous_rules, domains, instance, Cs, verbose=False):
    # Build new rules on top of the previous ones
    # The previous rules are not valid (i.d. they do not define causes)
    if previous_rules is None: return get_initial_rules(instance, domains)
    new_rules = set()

    # Iterate through the previous rules
    for rule in tqdm(previous_rules, disable=not verbose): # Complexity: O(1)
        C, W = get_sets(rule, instance)
        for feature, domain in enumerate(domains): # Complexity: O(n)
            # Don't consider features already in rule
            if feature in C|W:
                continue
                
            # Don't consider the rule if it is not minimal
            non_minimal_cause = any([c <= C|{feature} for c in Cs]) # Complexity O(n)
                

            # Add new rules with the feature
            for value in domain:
                # Check for minimality if we add a new variable to C
                if value != instance[feature] and non_minimal_cause:
                    continue
                # Build the rule
                new_rule = rule + ((feature, value),)
                # Add the new rule to the next rules
                new_rules.add(tuple(sorted(new_rule)))
    return sorted(new_rules)

def get_next_beams(non_causes, beam_size, Cs):
    non_causes = [v for v in non_causes if not any([c <= v[3] for c in Cs])]
    # Score and sort the remaining
    non_causes = sorted(non_causes, key=sort_key)
    # Filter the top-b
    if beam_size != -1:
        non_causes = non_causes[:beam_size]
    # Keep only the interventions to build next ones
    beams = [rule_value[0] for rule_value in non_causes]
    return beams

def split_rules(beams, cf_values, instance, epsilon):
    causes, non_causes = [], []
    for rule, (cf_state, cf_output, cf_score) in zip(beams, cf_values):
        C, W = get_sets(rule, instance)
        rule_value = (rule, cf_output, cf_score, C, W, cf_state)

        # Save causes and keep n best non-causes for next step
        if cf_output < epsilon: 
            causes.append(rule_value)
        else:
            non_causes.append(rule_value)
    return causes, non_causes

def check_early_stop(beams, early_stop, all_causes, max_time, init_time):
    if not len(beams): return True
    if early_stop and len(all_causes): return True
    if max_time is not None and time.time()-init_time > max_time: return True
    return False

def map_beams(beams, var_mapping, ref_w):
    if var_mapping is None:
        return beams
    sim_beams = [tuple([(var_mapping[var], value) for var, value in beam]) for beam in beams]
    if ref_w is not None:
        sim_beams = [beam + ref_w for beam in sim_beams]
    return sim_beams

def beam_search(
    instance, domains, simulation, variables, # SCM
    max_steps=5, beam_size=10, epsilon=.05, early_stop=True, max_time=None, # Parameters
    var_mapping=None, ref_w=tuple(), Cs=None, # Additional parameters when running for sub-instance
    verbose=0, 
    ):
    # verbose: 
    #  = 1 -> best cause at the end, tqdm for steps
    #  >= 2 -> removes step tqdm, adds step header + number of cause found + best and worse non causes
    #  >= 3 -> adds all causes + tqdm for get_rules
    
    all_causes = []
    init_time = time.time()
    if Cs is None: Cs = []
    if max_steps == -1 or max_steps is None: iterator = count(start=1, step=1)
    else: iterator = range(1,max_steps+1)

    for t in tqdm(iterator, disable=(verbose!=1)):
        # Render the step
        if verbose >= 2: print(f"{f'Step {t}':=^30}")
            
        # Create the rules for step t base on the ones from t-1, we use the initial ones if t==1
        beams = get_rules(beams if t > 1 else None, domains, instance, Cs, verbose >= 3)

        # Check for early stop
        if check_early_stop(beams, early_stop, all_causes, max_time, init_time):
            break
            
        # Render how many nodes will be evaluated
        if verbose >= 2: print(f"Evaluating {len(beams)} rules")
            
        # Evaluate the rules using the simulation 
        sim_beams = map_beams(beams, var_mapping, ref_w)
        cf_values = simulation(sim_beams)
        
        # Build the tuples of rule values
        causes, non_causes = split_rules(beams, cf_values, instance, epsilon)

        # Filter causes to keep only minimal ones and save them
        causes = filter_minimality(causes)
        all_causes += causes
        for rule_value in causes:
            Cs.append(rule_value[3])

        ### Build next beams
        beams = get_next_beams(non_causes, beam_size, Cs)
        
        # Render step output
        render_step(verbose, causes, non_causes, instance, variables)

    # Sort final rule set
    all_causes.sort(key=sort_key)

    # Render final result
    if verbose:
        print(f"----> Found {len(all_causes)} causes.")
        if all_causes:
            print(f"{'Overall best rule:':=^30}")
            show_rule(all_causes[0], variables)
        
    return all_causes
