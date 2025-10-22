
import numpy as np, time
from tqdm import tqdm
from collections import deque

from .base_algorithm import show_rules, get_initial_rules, get_rules, beam_search, get_rule_desc


def unmap_causes(causes, var_mapping, beams_w):
    if beams_w is None: beams_w = tuple()
    for i in range(len(causes)):
        cause = list(causes[i])
        cause[0] = [(var_mapping[dim], value) for dim, value in cause[0]] + list(beams_w)
        cause[3] = {var_mapping[dim] for dim in cause[3]}
        cause[4] = {var_mapping[dim] for dim in cause[4]} | {w[0] for w in beams_w}
        causes[i] = cause

def merge_set_lists(list1, list2):
    # Convert each set in the lists to a frozenset and add to a set to remove duplicates
    unique_frozensets = set()
    for s in list1:
        unique_frozensets.add(frozenset(s))
    for s in list2:
        unique_frozensets.add(frozenset(s))

    # Convert the frozensets back to sets
    merged_list = [set(fs) for fs in unique_frozensets]

    return merged_list

def map_cause_sets(Cs, var_mapping):
    mapped_Cs = []
    dim2index = {dim:i for i,dim in enumerate(var_mapping)}
    for C in Cs:
        mapped_C = set()
        for dim in C:
            if dim in dim2index:
                mapped_C.add(dim2index[dim])
            else:
                break
        else:
            mapped_Cs.append(mapped_C)
        
    return mapped_Cs

def unmap_cause_sets(Cs, var_mapping):
    return [{var_mapping[dim] for dim in C} for C in Cs]

def make_beam_search(instance, domains, simulation, variables, 
                     current_var_ids, Cs, beams_w,
                     **kargs):
    current_domains = [domains[var_id] for var_id in current_var_ids]
    current_variables = [variables[var_id] for var_id in current_var_ids]
    current_instance = [instance[var_id] for var_id in current_var_ids]
    
    mapped_Cs = map_cause_sets(Cs, current_var_ids)
    
    causes = beam_search(current_instance, current_domains, simulation, 
                                    current_variables, ref_w=beams_w,
                                    var_mapping=current_var_ids, Cs=mapped_Cs, **kargs)
    mapped_Cs = [v[3] for v in causes]
    
    unmap_causes(causes, current_var_ids, beams_w)
    new_Cs = unmap_cause_sets(mapped_Cs, current_var_ids)
    
    Cs = merge_set_lists(
        new_Cs, 
        Cs
    )
    
    return causes, Cs

def check_node_for_expansion(child_vars, visited, control):
    if not child_vars: return False
    if any([set(child_vars)<set(v) for v in visited]): return False
    if any([set(child_vars)<set(c) for c in control]): return False 
    return True

def expand_node(cause_vars, dag):
    children = []
    for n in range(2**len(cause_vars)-1):
        child = tuple()
        s = bin(n)[2:].zfill(len(cause_vars))
        for i,b in enumerate(s):
            if int(b):
                child += (cause_vars[i],)
            else:
                child += tuple(dag[cause_vars[i]])
        yield child

def iterative_identification(
    instance, domains, simulation, variables, dag, init_var_ids, 
    **kargs
):
    verbose = kargs.get("verbose", 0)
    early_stop = kargs.get("early_stop", True)
    queue = deque([(init_var_ids,None)])
    visited = set()
    current_depth = 0
    all_causes = []
    Cs = []
    while queue:
        # Set up node
        if verbose: print(f"{len(queue)=}")
        var_ids, beams_w = queue.popleft()
        visited.add(var_ids)
        if verbose: print(f"{var_ids=}, {beams_w=}")
        
        # Evaluate node
        causes, Cs = make_beam_search(instance, domains, simulation, variables, 
                                      var_ids, 
                                      beams_w=beams_w,
                                      Cs=Cs, **kargs)
        if early_stop and causes:
            return causes
        all_causes += causes
        current_depth += 1

        # Expand node
        control = set()
        for cause in causes:
            cause_vars = tuple(cause[3])
            beams_w = tuple([(dim, instance[dim]) for dim in cause[4]])
            for child_vars in expand_node(cause_vars, dag):
                if check_node_for_expansion(child_vars, visited, control):
                    if verbose: print(f"  Cause {get_rule_desc(cause, variables)} -> {child_vars} {beams_w}")
                    queue.append((child_vars,beams_w))
                    control.add(tuple(child_vars))
                
        
        if verbose: print("==========")
    return all_causes
