import numpy as np, time
from tqdm import tqdm
from collections import deque
from itertools import combinations, islice
from collections.abc import Iterable, Iterator

from .mbs import show_rules, get_initial_rules, get_rules, beam_search, get_rule_desc, is_minimal, remove_duplicates, minimal_merge


def merge_set_lists(list1:list[set], list2:list[set])->list[set]:
    # Convert each set in the lists to a frozenset and add to a set to remove duplicates
    unique_frozensets = set()
    for s in list1:
        unique_frozensets.add(frozenset(s))
    for s in list2:
        unique_frozensets.add(frozenset(s))
    # Convert the frozensets back to sets
    merged_list = [set(fs) for fs in unique_frozensets]
    return merged_list

def make_beam_search(v:list[object], D:list[list], V:list[str], K:'constraint', minimality:bool, 
                     **kargs) -> tuple['minimal MBS output', 'non-minimal MBS output']:
    # Use K to update D (using W_0) and build new V=W_0 U I
    I, W_0, R_C, r_C, R_W = K
    I = I|W_0
    R = tuple(r_C.items())
    R += tuple([(var, value) for var, value in zip(V,v) if var in R_W])
    D = [D[i] if var not in W_0 else [value] for i, (var, value) in enumerate(zip(V,v))]
    
    if minimality:
        E = beam_search(v=v, D=D, V=V, I=I, R=R, **kargs)
        full_E = None
    else:
        E, full_E = beam_search(v=v, D=D, V=V, I=I, R=R, minimality=False, **kargs)
        full_E = remove_duplicates(full_E)
    return E, full_E

def subsets(s:Iterable, n:int=None) -> Iterator[tuple]:
    s = list(s)
    if n is None:
        n = len(s)
    n = min(n + 1, len(s) + 1)
    for size in range(1, n):
        for subset in combinations(s, size):
            yield set(subset)

def CH(var:str, PA:dict[str:set[str]])->set[str]:
    return {child for child, parents in PA.items() if var in parents}

def CH_set(S:set[str], PA:dict[str:set[str]])->set[str]:
    chs = [CH(var, PA) for var in S]
    return set.union(*chs) if chs else set()

def PA_set(S:set[str], PA:dict[str:set[str]])->set[str]:
    pas = [set(PA[var]) for var in S]
    return set.union(*pas) if pas else set()

def desc(S:set[str], PA:dict[str:set[str]])->set[str]:
    ch = CH_set(S, PA)
    des = set()
    while ch:
        des |= ch
        ch = CH_set(ch, PA)
    return des

def anc(S:set[str], PA:dict[str:set[str]])->set[str]:
    pa = PA_set(S, PA)
    an = set()
    while pa:
        an |= pa
        pa = PA_set(pa, PA)
    return an

def expand(C:set, e:dict, W:set, S:set, PA:dict[str:set[str]]) -> tuple[set, set, set, dict,set]:
    I = PA_set(S, PA) - C
    W_0 = (desc(I, PA) - I) & (anc(C, PA) - C)
    R_C = C - S
    r_C = {var:e[var] for var in R_C}
    R_W = W | (desc(I, PA) - I - (desc(C, PA) | C | anc(C, PA)))
    return I, W_0, R_C, r_C, R_W

def check_memory(memory, K):
    I, W_0, R_C, r_C, R_W = K
    for I_ref, W_0_ref, R_C_ref, r_C_ref, R_W_ref in memory:
        W_0 -= W_0_ref
        R_C = {var for var in R_C if var not in R_C_ref or r_C[var] != r_C_ref[var]}
        R_W -= R_W_ref
        if I_ref >= I | W_0 | R_C | R_W:
            # print(I_ref, W_0_ref, R_C_ref, r_C_ref, R_W_ref, ">=", K)
            return False
    return True

def iterative_identification(v, D, simulation, V, dag, PA_T, cache_size=-1, **kargs):
    PA = dag
    minimality = max(map(len,D)) <= 2
    verbose = kargs.get("verbose", 0)
    early_stop = kargs.get("early_stop", False)
    kargs["simulation"] = simulation
    K_0 = (set(PA_T), set(), set(), dict(), set())
    queue = deque([K_0])
    cache = dict() if cache_size >= 0 else None
    actual_values = dict(zip(V,v))
    memory = []
    ret = []
    Cs = []
    while queue:
        # Set up node
        if verbose: print(f"{len(queue)=}")
        K = queue.popleft()
        while cache and len(cache) > cache_size:
            item = next(iter(cache))
            del cache[item]
        if cache_size >=0: 
            cache = dict(islice(cache.items(), cache_size))
        if verbose: print(f"{K=}")
        
        # Evaluate node
        E, full_E = make_beam_search(v, D, V, K, minimality, cache=cache, Cs=Cs, **kargs)
        
        Cs = merge_set_lists([e[3] for e in E], Cs)
        if early_stop and E: return E
        ret = minimal_merge(E, ret)

        # Expand node
        if not minimality: E = full_E
        for e in E:
            C, W = e[3], e[4]
            e = dict(e[0])
            for S in subsets(C):
                K = expand(C,dict(e),W,S, PA)
                if not len(K[0]) or not check_memory(memory, K): 
                    continue # no free variable for interventions or K already tested
                if verbose: print(f"  {C=} -> {S=} -> {K=}")
                queue.append(K)
                memory.append(K)
        
        if verbose: print("==========")
    return ret
