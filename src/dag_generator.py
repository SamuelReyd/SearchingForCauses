import math
from collections import defaultdict, deque
import numpy as np
from actualcauses import SCM, BaseNumpyModel
from dataclasses import dataclass, asdict
import networkx as nx
import matplotlib.pyplot as plt

T_LABEL = "T"
def ancestors_of(parents, node):
    seen, stack = set(), [node]
    while stack:
        x = stack.pop()
        for p in parents[x]:
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen

def restrict_to_ancestors(parents, target):
    """Keep only target and its ancestors; re-topo-sort with target last."""
    keep = ancestors_of(parents, target) | {target}
    sub = {v: [p for p in parents[v] if p in keep] for v in keep}
    # Kahn topological sort
    indeg = {v: 0 for v in sub}
    children = defaultdict(list)
    for v, ps in sub.items():
        for p in ps:
            children[p].append(v); indeg[v] += 1
    q = deque([v for v in sub if indeg[v] == 0]); order = []
    while q:
        x = q.popleft(); order.append(x)
        for c in children[x]:
            indeg[c] -= 1
            if indeg[c] == 0: q.append(c)
    order.remove(target)
    order.append(target)        # target last
    V = order
    parents_out = {v: sub[v] for v in V}
    sources = [v for v in V if not parents_out[v]]
    return V, parents_out, sources

def make_nodes(n):
    width = len(str(n - 1))
    return [f"n{i:0{width}d}" for i in range(n)]

# --------------------------------------------------------------------------- #
# Base DAG generators                                                         #
# --------------------------------------------------------------------------- #
def g_layer(L, w, p, rng):
    """Layered DAG: L layers of width w, edges layer->next w.p. p, single sink T."""
    layers = [[f"L{l}_{i}" for i in range(w)] for l in range(L)]
    parents = {v: [] for layer in layers for v in layer}
    for l in range(1, L):
        for v in layers[l]:
            ps = [u for u in layers[l-1] if rng.random() < p]
            if not ps:                                   # guarantee connectivity
                ps = [str(rng.choice(layers[l-1]))]
            parents[v] = ps
    parents[T_LABEL] = list(layers[-1])                  # sink over last layer
    return parents

def g_tree_shortcut(n, r, bias, rng):
    """In-tree rooted at T (reconvergence-free) + r shortcut edges making diamonds."""
    nodes = make_nodes(n-1) + [T_LABEL]    # T last in topo
    parents = {v: [] for v in nodes}
    # in-tree: every node i (>0) has exactly one child later in the order
    for i in range(len(nodes)-1):
        if bias == 0:
            child = rng.choice(nodes[i+1:])
        else:
            size = len(nodes) - (i + 1)
            weights = np.exp(bias * np.arange(1,size+1) / size)
            p = weights / sum(weights)
            child = rng.choice(nodes[i+1:], p=p)
        parents[child].append(nodes[i])
    added = 0
    tries = 0
    while added < r and tries < 50*r:  # add forward shortcut edges
        tries += 1
        a = rng.integers(0, len(nodes)-1)
        b = rng.integers(a+1, len(nodes))
        u, v = nodes[a], nodes[b]
        if u not in parents[v]:
            parents[v].append(u)
            added += 1
    if not parents[T_LABEL]: raise ValueError("T has no parents; increase n or decrease bias")
    return parents

def g_fanin(n, k, k_init, rng):
    """Topological order; each node draws up to k_max parents from earlier nodes."""
    nodes = make_nodes(n-1) + [T_LABEL]
    parents = {v: [] for v in nodes}
    for i in range(max(k, k_init) + 1, len(nodes)):
        parents[nodes[i]] = rng.choice(nodes[:i], k, replace=False).tolist()
    if nodes[-2] not in parents[T_LABEL]:
        parents[T_LABEL][0] = nodes[-2]
    return parents

def g_er(n, avg_degree, k_init, rng):
    """Erdos-Renyi DAG over a topological order at a target average degree."""
    nodes = make_nodes(n-1) + [T_LABEL]
    p = min(1.0, avg_degree / max(1, (n-1)/2))
    parents = {v: [] for v in nodes}
    for j in range(k_init, len(nodes)):
        for i in range(j):
            if rng.random() < p:
                parents[nodes[j]].append(nodes[i])
        if not parents[nodes[j]]:
            parents[nodes[j]] = [nodes[rng.integers(0, j)]]
    return parents

def g_bottleneck(d, w):
    """
    Deep 'master' source funnels (via necessary AND-paths) into m sufficient
    OR-parents of T.  No single direct parent of T cancels it (OR redundancy);
    the unique smallest cancelling intervention is the master, at depth d.
    Returns (parents, node_funcs) -- node_funcs encodes the logic.
    """
    master = "M"
    # d-1 chain nodes per branch from master to each pre-target p_i
    parents = {master: []}
    pre_targets = []
    node_funcs = {}
    for i in range(w):
        prev = master
        for lvl in range(1, d):                          # build a length-d chain
            node = f"c{i}_{lvl}"
            pad = f"pad{i}_{lvl}"                         # saturated enabler
            parents[pad] = []
            parents[node] = [prev, pad]
            node_funcs[node] = ("and",)                  # necessary: needs prev AND pad
            prev = node
        pi = f"p{i}"
        pad = f"padp{i}"
        parents[pad] = []
        parents[pi] = [prev, pad]
        node_funcs[pi] = ("and",)
        pre_targets.append(pi)
    parents[T_LABEL] = list(pre_targets)
    node_funcs[T_LABEL] = ("or",)
    return parents, node_funcs

dag_fnts = {
    "layer": g_layer,
    "shortcut": g_tree_shortcut,
    "fanin": g_fanin,
    "er": g_er,
    "bottleneck": g_bottleneck
}

# --------------------------------------------------------------------------- #
# Parameter classes                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class LayerParams:
    L: int = 4
    w: int = 4
    p: int = .5

@dataclass 
class TreeParams:
    n: int = 17
    r: int = 10
    bias: float = 0.0

@dataclass
class FaninParams:
    n: int = 17
    k: int = 3
    k_init: int = 5

@dataclass
class ERParams:
    n: int = 17
    avg_degree: int = 3
    k_init: int = 5

@dataclass
class BottleneckParams:
    d: int = 4
    w: int = 4

# --------------------------------------------------------------------------- #
# Structure-faithful SCM (canonical = fractional threshold)                   #
# --------------------------------------------------------------------------- #
def make_node_funcs_threshold(dag, theta):
    return {v: ("threshold", theta) for v, pa in dag.items() if pa}

def make_node_funcs_neg_threshold(dag, theta):
    """Node is 1 when proportion of positive parents < |theta|."""
    return {v: ("neg_threshold", abs(theta)) for v, pa in dag.items() if pa}
 
def make_node_funcs_parity(dag):
    """XOR over parents."""
    return {v: ("parity",) for v, pa in dag.items() if pa}

def make_F(dag, theta):
    """Dispatch to the right node-function factory based on theta value."""
    if theta == "parity":
        return make_node_funcs_parity(dag)
    elif theta >= 0:
        return make_node_funcs_threshold(dag, theta)
    else:
        return make_node_funcs_neg_threshold(dag, abs(theta))

class StructModel(BaseNumpyModel):
    """Per-node functions over parents. Sources read their value from context u."""
    def __init__(self, dag, F, **kw):
        self.dag = dag
        self.V = list(dag.keys())
        self.U = [v for v, ch in dag.items() if not ch]
        self.F = F
        self.order = [v for v in self.V if v not in self.U]       # non-sources, topo
        super().__init__(self.V, **kw)

    def simulate(self, u):
        for i, s in enumerate(self.U):
            self[s] = u[i]
        for var in self.order:
            pa  = self.dag[var]
            cols = [self[p].astype(int) for p in pa]
            kind = self.F[var]
            if kind[0] == "threshold":
                theta = kind[1]
                psum  = sum(cols)
                thr   = max(1, math.ceil(theta * len(pa)))
                val   = (psum >= thr)
            elif kind[0] == "neg_threshold":
                theta = kind[1]
                psum  = sum(cols)
                thr   = max(1, math.ceil(theta * len(pa)))
                val   = (psum < thr)
            elif kind[0] == "parity":
                val = cols[0].astype(bool)
                for c in cols[1:]:
                    val = val ^ c.astype(bool)
            elif kind[0] == "and":
                val = cols[0].astype(bool)
                for c in cols[1:]:
                    val = val & c.astype(bool)
            elif kind[0] == "or":
                val = cols[0].astype(bool)
                for c in cols[1:]:
                    val = val | c.astype(bool)
            else:
                raise ValueError(kind)
            self[var] = val.astype(self.dtype)

    def to_json(self):
        return {
            "dag": self.dag,
            "F": self.F
        }
    
    @staticmethod
    def from_json(json_dict, **kw):
        return StructModel(json_dict["dag"], json_dict["F"], **kw)

def build_model(dag, F):
    return StructModel(dag, F)

def build_scm(dag, u, F, v=None):
    model = build_model(dag, F)
    return SCM(V=model.V, U=model.U, D=[0, 1], u=list(u),
               model=model, dag=dag, v=v)

# --------------------------------------------------------------------------- #
# 3. context sampling + causal proxies (confirm the knob landed)              #
# --------------------------------------------------------------------------- #
def pivotal_parents(scm, var):
    """Parents whose unilateral flip changes `var` in the actual context."""
    base = dict(zip(scm.V, scm.v))
    piv = []
    for p in scm.dag[var]:
        e = [(p, 1 - base[p])]
        s = dict(zip(scm.V, scm.model(scm.u, e)))
        if s[var] != base[var]:
            piv.append(p)
    return piv

def mean_causal_indegree(scm):
    nz = [scm.dag[v] for v in scm.V if scm.dag[v]]
    if not nz: return 0.0
    return np.mean([len(pivotal_parents(scm, v)) for v in scm.V if scm.dag[v]])

def count_reconvergences(parents):
    """Nodes with >=2 parents whose ancestor sets (incl. selves) intersect."""
    anc = {v: ancestors_of(parents, v) | {v} for v in parents}
    cnt = 0
    for v, ps in parents.items():
        coupled = any(anc[a] & anc[b]
                      for i, a in enumerate(ps) for b in ps[i+1:])
        cnt += int(coupled)
    return cnt

def realized_cause_depth(scm):
    """Depth-to-T of the shallowest variable of the smallest exact cause."""
    depth = {scm.V[-1]: 0}
    for v in reversed(scm.V):                              # children known first
        for p in scm.dag[v]:
            depth[p] = max(depth.get(p, 0), depth[v] + 1)
    if not scm.causes: return None
    smallest = min(scm.causes, key=len)
    return min(depth[x] for x in smallest)                 # how deep the cause sits

# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def show_layer_scm(scm):
    ids = np.argsort(scm.U)
    for i in ids:
        print(f"{scm.U[i]}: {scm.u[i]}")
    ids = np.argsort(scm.V)
    for i in ids:
        print(f"{scm.V[i]}: {scm.v[i]}")

def show_dag(dag):
    for v in sorted(dag.keys()):
        print(f"{v}: {sorted(dag[v])}")

def show_scm(scm):
    print("DAG:")
    show_dag(scm.dag)
    print("Context u:")
    for i, s in enumerate(scm.U):
        print(f"{s}: {scm.u[i]}")
    print("Realization v:")
    for i, s in enumerate(scm.V):
        print(f"{s}: {int(scm.v[i])}")

def render_dag(dag, figsize=(6, 4), node_size=500, font_size=10):
    G = nx.DiGraph()
    for v, ps in dag.items():
        for p in ps:
            G.add_edge(p, v)
    pos = nx.spring_layout(G)
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=True, arrows=True,
            node_size=node_size, font_size=font_size)
    plt.show()

if __name__ == "__main__":
    rng = np.random.default_rng(40)

    # dag = g_tree_shortcut(n=20, r=10, rng=rng)
    # u = [0,1,0,1,0,1,0]
    # scm = build_scm(dag, u=u, F=make_F(dag, 0.5))
    # show_scm(scm)
    # scm.find_causes(True, beam_size=-1, max_steps=-1)
    # scm.show_identification_result()


    dag = g_tree_shortcut(n=50, r=3, bias=-3, rng=rng)
    # show_dag(dag)
    u_nb = len([v for v, ch in dag.items() if not ch])
    u = np.random.randint(0, 2, size=u_nb).tolist()
    scm = build_scm(dag, u=u, F=make_F(dag, 0.5))
    show_scm(scm)
    scm.find_causes(True, beam_size=-1, max_steps=-1)
    scm.show_identification_result()