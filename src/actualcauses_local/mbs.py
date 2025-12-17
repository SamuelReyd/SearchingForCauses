import numpy as np, bisect, time
from tqdm import tqdm
from itertools import count
from collections import defaultdict

def render_step(verbose, causes, non_causes, Cs):
    non_causes = [v for v in non_causes if not any([c <= v[3] for c in Cs])]
    if len(non_causes):
        if verbose == 2:
            non_causes = sorted(non_causes, key=sort_key)
            print("Number of causes found:", len(causes))
            print("Number of non-causes remaining:", len(non_causes))
            print("Best non-cause:")
            show_rule(non_causes[0])
            print("Worst non-cause:")
            show_rule(non_causes[-1])
        if verbose >= 3:
            print("Causes found this step:")
            show_rules(causes)
            print("Rules passed for next step:")
            show_rules(non_causes)
    else:
        if verbose >= 2:
            print("No rule available")

def get_sets(rule, actual_values):
    C = set()
    W = set()
    for feature, value in rule:
        if actual_values[feature] != value:
            C.add(feature)
        else:
            W.add(feature)
    return C, W

def sort_key(rule_values):
    _, rule_output, rule_score, C, W = rule_values
    return (rule_score, len(C), C, len(W), W)

def format_value(value):
    if isinstance(value, float):
        return f"{float(value):.2f}"
    elif isinstance(value, int):
        return f"{int(value)}"
    else:
        return f"{value}"

def get_rule_desc(rule_values, show_score=False):
    rule, output, score, C, W = rule_values
    dim2value = dict(rule)
    C = {c:format_value(dim2value[c]) for c in C}
    W = {w:format_value(dim2value[w]) for w in W}
    if show_score: 
        if isinstance(score, tuple):
            return f"{C=}, {W=}, {output=}, {score=}"
        return f"{C=}, {W=}, {output=}, {score=:.3f}"
    return f"{C=}, {W=}"

def show_rule(rule_values):
    print(get_rule_desc(rule_values, True))
    
def show_rules(rule_values):
    for r_values in rule_values:
        show_rule(r_values)

def is_minimal(e, E):
    return not any([other[3] < e[3] for other in E])

def minimal_merge(E1, E2):
    E = [e for e in E1 if is_minimal(e, E2)] + [e for e in E2 if is_minimal(e, E1)]
    return remove_duplicates(E)

def filter_minimality(E):
    # Remove causes with strict subsets that are causes
    E = [e for e in E if is_minimal(e, E)]

    # If there are equalities, keep the ones with smallest W and best score
    E = remove_duplicates(E)
    return E

def remove_duplicates(E):
    Cs = defaultdict(lambda: [])
    for e in E:
        Cs[tuple(e[3])].append(e)
    causes = []
    for cands in Cs.values():
        best = min(cands, key=lambda e: (len(e[4]),e[2]))
        causes.append(best)
    return causes
    

def get_initial_rules(V, D, v):
    rules = []
    for actual_value, feature, domain in zip(v, V, D):
        for value in domain:
            if actual_value != value:
                rules.append(((feature, value),))
    return rules

def filter_instance(V, D, v, I):
    return zip(*[(variable, domain, value) for variable, domain, value in zip(V, D, v) if variable in I])

def get_rules(previous_rules, V, D, v, actual_values, Cs=[], R=[], verbose=False):
    # Build new rules on top of the previous ones
    # The previous rules are not valid (i.d. they do not define causes)
    # if verbose: print(previous_rules, V, D, v, Cs, actual_values)
    if previous_rules is None: 
        rules = get_initial_rules(V, D, v)
        return [beam + R for beam in sorted(rules)]
    new_rules = set()

    # Iterate through the previous rules
    for rule in tqdm(previous_rules, disable=not verbose): # Complexity: O(1)
        C, W = get_sets(rule, actual_values)
        for actual_value, feature, domain in zip(v, V, D): # Complexity: O(n)
            # Don't consider features already in rule
            if feature in C|W:
                continue
                
            # Don't consider the rule if it is not minimal
            non_minimal_c = any([c <= C|{feature} for c in Cs]) # Complexity O(n)
            

            # Add new rules with the feature
            for value in domain:
                # Check for minimality if we add a new variable to C
                if value != actual_value and non_minimal_c:
                    continue
                # Build the rule
                new_rule = rule + ((feature, value),)
                # Add the new rule to the next rules
                new_rules.add(tuple(sorted(new_rule)))
    return [beam + R for beam in sorted(new_rules)]

def get_next_beam(non_causes, beam_size, Cs):
    non_causes = [v for v in non_causes if not any([c <= v[3] for c in Cs])]
    # Score and sort the remaining
    non_causes = sorted(non_causes, key=sort_key)
    # Filter the top-b
    if beam_size != -1:
        non_causes = non_causes[:beam_size]
    # Keep only the interventions to build next ones
    beam = [rule_value[0] for rule_value in non_causes]
    return beam

def split_rules(beam, cf_values, actual_values, epsilon):
    causes, non_causes = [], []
    for rule, (cf_output, cf_score) in zip(beam, cf_values):
        C, W = get_sets(rule, actual_values)
        rule_value = (rule, float(cf_output), float(cf_score), C, W)

        # Save causes and keep n best non-causes for next step
        if cf_output < epsilon: 
            causes.append(rule_value)
        else:
            non_causes.append(rule_value)
    return causes, non_causes

def check_early_stop(beam, early_stop, all_causes, max_time, init_time):
    if not len(beam): return True
    if early_stop and len(all_causes): return True
    if max_time is not None and time.time()-init_time > max_time: return True
    return False

def do_simulation(simulation, cache, beam):
    if cache is None: return simulation(beam)
    cached_results = []
    non_cached_beam = []
    for e in beam:
        if e in cache:
            cached_results.append(cache[e])
        else:
            non_cached_beam.append(e)
    cf_values = simulation(non_cached_beam)
    cache |= dict(zip(non_cached_beam, cf_values))
    return cached_results + cf_values

# def test_empty(simulation):
#     beam = [()]
#     cf_values = simulation(beam)

def beam_search(
    v, D, simulation, V, # SCM
    max_steps=5, beam_size=10, epsilon=.05, early_stop=False, max_time=None, # Parameters
    R=None, Cs=None, cache=None, minimality=True, I=None, # Additional parameters when running for sub-instance
    verbose=0, 
    ):
    # verbose: 
    #  = 1 -> best cause at the end, tqdm for steps
    #  >= 2 -> removes step tqdm, adds step header + number of cause found + best and worse non causes
    #  >= 3 -> adds all causes + tqdm for get_rules
    
    all_causes = []
    if R is None: R = tuple()
    actual_values = dict(zip(V, v))
    if I is not None: V, D, v = filter_instance(V,D, v, I)
    # print(I, V)
    if not minimality: full_interventions = []
    init_time = time.time()
    beam = None
    if Cs is None: Cs = []
    if max_steps == -1 or max_steps is None: iterator = count(start=1, step=1)
    else: iterator = range(1,max_steps+1)

    # if test_empty(simulation): iterator = range(0,-1)

    for t in tqdm(iterator, disable=(verbose!=1)):
        # Render the step
        if verbose >= 2: print(f"{f'Step {t}':=^30}")
            
        # Create the rules for step t base on the ones from t-1, we use the initial ones if t==1
        beam = get_rules(beam, V, D, v, actual_values, 
                          Cs=Cs if minimality else [], 
                          R=R, verbose=verbose >= 3)

        # Check for early stop
        if check_early_stop(beam, early_stop, all_causes, max_time, init_time):
            break
            
        # Render how many nodes will be evaluated
        if verbose >= 2: print(f"Evaluating {len(beam)} rules")

        # Evaluate the rules using the simulation 
        cf_values = do_simulation(simulation, cache, beam)
        # Build the tuples of rule values
        AC2, non_AC2 = split_rules(beam, cf_values, actual_values, epsilon)

        # Filter causes to keep only minimal ones
        causes = filter_minimality(AC2)

        # Save minimal causes
        if minimality:
            all_causes += causes
        # Search for all counterfactual interventions:
        else:
            # Filter non-minimal propagated causes
            all_causes = minimal_merge(all_causes, causes)
            # Save all counterfactual interventions
            full_interventions += AC2
            # Remove duplicates but not non-minimal interventions
            # full_interventions = remove_duplicates(full_interventions)

        # Save the cause sets
        for rule_value in causes:
            Cs.append(rule_value[3])

        # Build next beam
        if minimality:
            beam = get_next_beam(non_AC2, beam_size, Cs)
        else:
            beam = get_next_beam(non_AC2 + AC2, beam_size, [])
        # Render step output
        render_step(verbose, causes, non_AC2, Cs)

    # Render final result
    if verbose:
        print(f"----> Found {len(all_causes)} causes.")
    if not minimality: return all_causes, full_interventions
    return all_causes
