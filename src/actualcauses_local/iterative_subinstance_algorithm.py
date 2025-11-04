
import numpy as np, time
from tqdm import tqdm
from collections import deque
from itertools import islice

from .base_algorithm import show_rules, get_initial_rules, get_rules, beam_search, get_rule_desc, is_minimal


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

def make_beam_search(v, D, simulation, V, 
                     I, Cs, W_R,
                     **kargs):
    W_R = tuple([(var, value) for var, value in zip(V,v) if var in W_R])
    D, V, v = zip(*[(domain, variable, value) for domain, variable, value in zip(D,V,v) if variable in I])
    
    E = beam_search(v, D, simulation, V, W_R=W_R, Cs=Cs, **kargs)
    
    Cs = merge_set_lists(
        [e[3] for e in E], 
        Cs
    )
    
    return E, Cs

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
        yield set(closure)

def CH(var, PA):
    return {child for child, parents in PA.items() if var in parents}

def CH_set(S, PA):
    return set.union(*[CH(var, PA) for var in S])

def W_R_compl(C, back, PA, boolean):
    excl = set(C) | set(back)
    anc = CH_set(back, PA) - excl
    if boolean:
        return anc - CH_set(C, PA)
    ch = anc.copy()
    while ch:
        ch = CH_set(ch) - excl
        anc |= ch
    return anc

def minimal_merge(E1, E2):
    return [e for e in E1 if is_minimal(e, E2)] + [e for e in E2 if is_minimal(e, E1)]


def iterative_identification(
    v, D, simulation, V, dag, PA_T, cache_size=10000,
    **kargs
):
    PA = dag
    boolean = max(map(len,D)) <= 2
    verbose = kargs.get("verbose", 0)
    early_stop = kargs.get("early_stop", False)
    queue = deque([(PA_T,tuple())]) # I, W_R, cache
    visited = set()
    ret = []
    Cs = []
    while queue:
        # Set up node
        if verbose: print(f"{len(queue)=}")
        # I, W_R, cache = queue.popleft()
        I, W_R = queue.popleft()
        # cache = dict(islice(cache.items(), cache_size))
        visited.add(tuple(I))
        if verbose: print(f"{I=}, {W_R=}")
        
        # Evaluate node
        E, Cs = make_beam_search(v, D, simulation, V, I, cache=None,
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
                    if verbose: print(f"  {C=} -> {next_I=} {next_W_R=}")
                        
                    # queue.append((next_I,next_W_R,cache))
                    queue.append((next_I,next_W_R))
                    control.add(tuple(next_I))
                
        
        if verbose: print("==========")
    return ret
