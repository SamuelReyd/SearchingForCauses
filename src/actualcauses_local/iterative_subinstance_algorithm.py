
import numpy as np, time
from tqdm import tqdm
from collections import deque

from .base_algorithm import show_rules, get_initial_rules, get_rules, beam_search, get_rule_desc, is_minimal


def unmap_causes(causes, var_mapping, W_R):
    for i in range(len(causes)):
        cause = list(causes[i])
        cause[0] = [(var_mapping[dim], value) for dim, value in cause[0]] + list(W_R)
        cause[3] = {var_mapping[dim] for dim in cause[3]}
        cause[4] = {var_mapping[dim] for dim in cause[4]} | {w[0] for w in W_R}
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
                     current_var_ids, Cs, W_R,
                     **kargs):
    current_domains = [domains[var_id] for var_id in current_var_ids]
    current_variables = [variables[var_id] for var_id in current_var_ids]
    current_instance = [instance[var_id] for var_id in current_var_ids]
    
    mapped_Cs = map_cause_sets(Cs, current_var_ids)
    
    causes = beam_search(current_instance, current_domains, simulation, 
                                    current_variables, W_R=W_R,
                                    var_mapping=current_var_ids, Cs=mapped_Cs, **kargs)
    mapped_Cs = [v[3] for v in causes]
    
    unmap_causes(causes, current_var_ids, W_R)
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

def backtrack_closure(C, PA):
    C = tuple(C)
    for n in range(2**len(C)-1):
        closure = tuple()
        s = bin(n)[2:].zfill(len(C))
        for i,b in enumerate(s):
            if int(b):
                closure += (C[i],)
            else:
                closure += tuple(PA[C[i]])
        yield tuple(set(closure))

def CH(var, PA):
    return {i for i, parents in enumerate(PA) if var in parents}

def CH_set(S, PA):
    return set.union(*[CH(var, PA) for var in S])

def W_R_compl(C, back, PA, boolean):
    C = set(C)
    anc = CH_set(back, PA) - C
    if boolean:
        return anc
    ch = anc.copy()
    while ch:
        ch = CH_set(ch) - C
        anc |= ch
    return anc


def minimal_merge(E1, E2):
    return [e for e in E1 if is_minimal(e, E2)] + [e for e in E2 if is_minimal(e, E1)]

def show_info(C, V, next_I, next_W_R, verbose):
    if verbose: 
        cause_repr = get_rule_desc(C, V)
        child_repr = [V[i] for i in next_I]
        W_repr = [V[i[0]] for i in next_W_R]
        print(f"  Cause {cause_repr} -> Exp:{child_repr} W:{W_repr}")

def iterative_identification(
    v, D, simulation, V, dag, PA_T, 
    **kargs
):
    PA = dag
    boolean = max(map(len,D)) <= 2
    verbose = kargs.get("verbose", 0)
    early_stop = kargs.get("early_stop", False)
    queue = deque([(PA_T,tuple())])
    visited = set()
    ret = []
    Cs = []
    while queue:
        # Set up node
        if verbose: print(f"{len(queue)=}")
        I, W_R = queue.popleft()
        visited.add(tuple(I))
        if verbose: print(f"{I=}, {W_R=}")
        
        # Evaluate node
        E, Cs = make_beam_search(v, D, simulation, V, I, 
                                 minimality=boolean, W_R=W_R, 
                                 Cs=Cs, **kargs)
        if early_stop and E:
            return E
        ret = minimal_merge(E, ret)

        # Expand node
        control = set()
        for e in E:
            C = e[3]
            for next_I in backtrack_closure(C, PA):
                if check_node_for_expansion(next_I, visited, control):
                    next_W_R = W_R_compl(C, next_I, PA, boolean) | e[4]
                    next_W_R = tuple([(dim, v[dim]) for dim in next_W_R])
                    show_info(e, V, next_I, next_W_R, verbose)
                        
                    queue.append((next_I,next_W_R))
                    control.add(tuple(next_I))
                
        
        if verbose: print("==========")
    return ret
