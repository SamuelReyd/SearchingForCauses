
import numpy as np, time
from tqdm import tqdm
from collections import deque
from itertools import combinations, islice

from .base_algorithm import show_rules, get_initial_rules, get_rules, beam_search, get_rule_desc, is_minimal, remove_duplicates, minimal_merge


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

def make_beam_search(v, D, simulation, V, I, Cs, R, minimality, **kargs):
    # W_R = tuple([(var, value) for var, value in zip(V,v) if var in W_R])
    # D, V, v = zip(*[(domain, variable, value) for domain, variable, value in zip(D,V,v) if variable in I])
    if minimality:
        E = beam_search(v, D, simulation, V, I=I, R=R, Cs=Cs, **kargs)
        full_E = None
    else:
        E, full_E = beam_search(v, D, simulation, V, I=I, R=R, Cs=Cs, 
                                minimality=minimality, **kargs)
    
    Cs = merge_set_lists([e[3] for e in E], Cs)
    return E, Cs, full_E

def check_node_for_expansion(child_vars, visited, control):
    if not child_vars: return False
    if any([set(child_vars)<=set(v) for v in visited]): return False
    if any([set(child_vars)<=set(c) for c in control]): return False
    return True

# def backtrack_closure(C, PA):
#     C = tuple(C)
#     for n in range(2**len(C)-1):
#         closure = tuple()
#         s = bin(n)[2:].zfill(len(C))
#         for i,b in enumerate(s):
#             if int(b):
#                 closure += (C[i],)
#             else:
#                 closure += tuple(PA[C[i]])
#         yield set(closure)

def subsets(s, n=None):
    s = list(s)
    if n is None:
        n = len(s)
    n = min(n + 1, len(s) + 1)
    for size in range(1, n):
        for subset in combinations(s, size):
            yield set(subset)

def CH(var, PA):
    return {child for child, parents in PA.items() if var in parents}

def CH_set(S, PA):
    return set.union(*[CH(var, PA) for var in S])

def anc(S, PA):
    ch = CH_set(S, PA)
    anc = set()
    while ch:
        anc |= ch
        ch = CH_set(ch, PA)
    return anc

def W_R_compl(C, back, PA, boolean):
    excl = set(C) | set(back)
    if boolean:
        return CH_set(back - C, PA) - excl - CH_set(C, PA)
    return anc(back - C, PA) - excl - anc(C, PA)


def iterative_identification(v, D, simulation, V, dag, PA_T, cache_size=-1,**kargs):
    PA = dag
    boolean = max(map(len,D)) <= 2
    # boolean = True
    verbose = kargs.get("verbose", 0)
    early_stop = kargs.get("early_stop", False)
    queue = deque([(PA_T,tuple())])
    cache = dict() if cache_size >= 0 else None
    visited = set()
    actual_values = dict(zip(V,v))
    ret = []
    Cs = []
    while queue:
        # Set up node
        if verbose: print(f"{len(queue)=}")
        I, R = queue.popleft()
        while cache and len(cache) > cache_size:
            item = next(iter(cache))
            del cache[item]
        if cache_size >=0: 
            cache = dict(islice(cache.items(), cache_size))
        visited.add(tuple(I))
        if verbose: print(f"{I=}, {R=}")
        
        # Evaluate node
        E, Cs, full_E = make_beam_search(v, D, simulation, V, I=I, cache=cache,
                                         minimality=boolean, R=R, 
                                         Cs=Cs, **kargs)
        if early_stop and E: return E
        ret = minimal_merge(E, ret)

        # Expand node
        control = set()
        if not boolean: E = full_E
        for e in E:
            C, W = e[3], e[4]
            e = dict(e[0])
            for S in subsets(C):
                next_I = set.union(*[set(PA[var]) for var in S]) - C
                if not len(next_I): continue
                block = C-next_I-S
                C_R = tuple([(v,e[v]) for v in block])
                if check_node_for_expansion(next_I, visited, control):
                    W_R = W_R_compl(C, next_I, PA, boolean) | W
                    if verbose: print(f"  {C=} -> {S=} {next_I=} {W_R=} {C_R=}")
                    W_R = tuple([(v,actual_values[v]) for v in W_R])
                    queue.append((next_I,W_R+C_R))
                    control.add(tuple(next_I))
                
        
        if verbose: print("==========")
    return ret
