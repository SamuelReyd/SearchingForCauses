import random
from itertools import product
import csv 
import os 
import time 
import pandas as pd
from tqdm import tqdm

import numpy as np

from actualcauses import beam_search, iterative_identification, lucb

# =========================================================================== #
#                                                                             #
#   random SCMs, the corpus, graph perturbations, the stochastic variant      #
#                                                                             #
# =========================================================================== #
class RandomSCM:
    """An acyclic SCM over finite domains, in a fixed context.

    `simulation` is the counterfactual evaluator the search code expects: it
    returns one `(output, score)` pair per intervention, `output` being 0.0 when
    the intervention changes the target and 1.0 otherwise, so that "output below
    epsilon" means "the target is cancelled" for any epsilon in (0, 1).  The
    score carries no heuristic signal (score = output), which is deliberate: the
    study measures search structure, not heuristic quality.
    """
    def __init__(self, variables, target, domains, parents, mechanisms, context):
        self.variables = list(variables)
        self.target = target
        self.domains = dict(domains)
        self.parents = {variable: list(pa) for variable, pa in parents.items()}
        self.mechanisms = dict(mechanisms)
        self.context = dict(context)
        self.n_calls = 0
        self._order = self.variables + [self.target]
        # memoised: this only speeds up the brute-force enumeration, the counter
        # is incremented before the cache is consulted
        self._cache = {}
        self.actual_values = self.evaluate()

    def evaluate(self, intervention=None) -> dict:
        intervention = {} if intervention is None else dict(intervention)
        key = tuple(sorted(intervention.items()))
        if key in self._cache:
            return self._cache[key]
        values = {}
        for variable in self._order:
            if variable in intervention:
                values[variable] = intervention[variable]
            elif not self.parents[variable]:
                values[variable] = self.context[variable]
            else:
                parent_values = tuple(values[p] for p in self.parents[variable])
                values[variable] = self.mechanisms[variable][parent_values]
        self._cache[key] = values
        return values

    def cancels(self, intervention) -> bool:
        return self.evaluate(intervention)[self.target] != self.actual_values[self.target]

    def simulation(self, rules) -> list:
        results = []
        for rule in rules:
            self.n_calls += 1
            output = 0.0 if self.cancels(rule) else 1.0
            results.append((output, output))
        return results

    def reset_counter(self) -> None:
        self.n_calls = 0

    def isi_inputs(self) -> tuple:
        """`(v, D, simulation, V, dag, PA_T)`; V excludes the target, dag includes it."""
        v = [self.actual_values[variable] for variable in self.variables]
        D = [list(self.domains[variable]) for variable in self.variables]
        dag = {variable: list(pa) for variable, pa in self.parents.items()}
        return v, D, self.simulation, list(self.variables), dag, list(self.parents[self.target])


def generate_random_scm(n_variables=6, domain_size=2, edge_probability=0.5,
                        seed=None, conjunction_bias=0.0) -> RandomSCM:
    """Draw a random acyclic SCM together with a context.

    `conjunction_bias` is the probability that a mechanism is a *coincidence
    gate* -- one value on exactly one combination of its parents -- instead of a
    uniform draw.  Uniform mechanisms almost always give causes of size one;
    coincidence gates give causes as large as the in-degree.
    """
    rng = random.Random(seed)
    names = [f"X{index}" for index in range(n_variables)]
    target = "T"

    while True:
        ordered = names + [target]
        parents = {}
        for position, variable in enumerate(ordered):
            parents[variable] = [candidate for candidate in ordered[:position]
                                 if rng.random() < edge_probability]
        if parents[target]:
            break

    relevant = {target}
    changed = True
    while changed:
        changed = False
        for variable in list(relevant):
            for parent in parents[variable]:
                if parent not in relevant:
                    relevant.add(parent)
                    changed = True
    ordered = [variable for variable in ordered if variable in relevant]
    parents = {variable: parents[variable] for variable in ordered}

    values = list(range(domain_size))
    domains = {variable: list(values) for variable in ordered}
    mechanisms, context = {}, {}
    for variable in ordered:
        if parents[variable]:
            if rng.random() < conjunction_bias:
                pattern = tuple(rng.choice(values) for _ in parents[variable])
                hit, miss = rng.sample(values, 2) if domain_size > 1 else (0, 0)
                mechanisms[variable] = {
                    parent_values: (hit if parent_values == pattern else miss)
                    for parent_values in product(values, repeat=len(parents[variable]))
                }
            else:
                mechanisms[variable] = {
                    parent_values: rng.choice(values)
                    for parent_values in product(values, repeat=len(parents[variable]))
                }
        else:
            context[variable] = rng.choice(values)

    endogenous = [variable for variable in ordered if variable != target]
    return RandomSCM(endogenous, target, domains, parents, mechanisms, context)

# --------------------------------------------------------------------------- #
# graph misspecification
# --------------------------------------------------------------------------- #
def inflate_parents(scm, n_extra: int, seed: int) -> tuple:
    """The DAG with `n_extra` spurious parents per variable.

    Extra parents are drawn among the variables that precede the child in the
    topological order, so the graph stays acyclic.  Mechanisms are untouched:
    only the graph handed to the algorithm is wrong.
    """
    rng = random.Random(seed)
    order = scm.variables + [scm.target]
    position = {variable: index for index, variable in enumerate(order)}
    dag = {}
    for variable in order:
        parents = list(scm.parents[variable])
        candidates = [other for other in order[:position[variable]]
                      if other not in parents]
        rng.shuffle(candidates)
        dag[variable] = parents + candidates[:n_extra]
    return dag, list(dag[scm.target])


def deflate_parents(scm, n_missing: int, seed: int) -> tuple:
    """The DAG with up to `n_missing` parents removed per variable.

    The mirror image of `inflate_parents`, and the case that can actually hurt:
    ISI builds its free instances from Pa(S), so an edge missing from the graph
    is a variable the search never considers.  The target keeps at least one
    parent, otherwise the root subproblem is empty and nothing is searched at
    all, which would measure a degenerate case rather than the effect of the
    error.
    """
    rng = random.Random(seed)
    dag = {}
    for variable in scm.variables + [scm.target]:
        parents = list(scm.parents[variable])
        floor = 1 if variable == scm.target else 0
        n_drop = min(n_missing, max(0, len(parents) - floor))
        if n_drop:
            dropped = set(rng.sample(parents, n_drop))
            parents = [p for p in parents if p not in dropped]
        dag[variable] = parents
    return dag, list(dag[scm.target])


# --------------------------------------------------------------------------- #
# the stochastic variant
# --------------------------------------------------------------------------- #
class NoisyRandomSCM:
    """A random SCM whose mechanisms misfire independently at each evaluation.

    Every variable with parents takes, with probability `noise`, a value drawn
    uniformly among the other values of its domain instead of the one its
    mechanism prescribes.  One *sample* is one call, so `n_calls` counts samples
    and stays comparable with the deterministic study.

    `phi` is the probability that the target keeps its actual value, so an
    intervention cancels when `phi < a`.  The reference causes remain those of
    the underlying deterministic model: the question is whether the algorithms
    recover the causes of the system from noisy observations of it.
    """

    def __init__(self, scm: RandomSCM, noise: float, a: float, seed: int = 0):
        self.scm = scm
        self.noise = noise
        self.a = a
        self.rng = random.Random(seed)
        self.variables = scm.variables
        self.target = scm.target
        self.domains = scm.domains
        self.parents = scm.parents
        self.actual_values = scm.actual_values
        self.n_calls = 0
        self.decisions = []           # (estimated cancel, true cancel) per rule

    def reset_counter(self) -> None:
        self.n_calls = 0
        self.decisions = []

    def sample_once(self, intervention) -> float:
        """1.0 if the target keeps its actual value under one noisy run."""
        self.n_calls += 1
        intervention = dict(intervention)
        values = {}
        for variable in self.scm._order:
            if variable in intervention:
                values[variable] = intervention[variable]
            elif not self.scm.parents[variable]:
                values[variable] = self.scm.context[variable]
            else:
                parent_values = tuple(values[p] for p in self.scm.parents[variable])
                value = self.scm.mechanisms[variable][parent_values]
                if self.rng.random() < self.noise:
                    others = [d for d in self.scm.domains[variable] if d != value]
                    if others:
                        value = self.rng.choice(others)
                values[variable] = value
        return 0.0 if values[self.target] != self.actual_values[self.target] else 1.0

    def evaluator(self, E, bs):
        """LUCB's sampler: `bs` samples of `(phi, psi)` for each rule of E."""
        out = np.empty((len(E) * bs, 2), dtype=float)
        index = 0
        for rule in E:
            for _ in range(bs):
                value = self.sample_once(rule)
                out[index] = (value, value)
                index += 1
        return out

    def _record(self, rules, phis) -> None:
        for rule, phi in zip(rules, phis):
            self.decisions.append((bool(phi < self.a), bool(self.scm.cancels(rule))))

    def make_lucb_simulation(self, beam_size, max_iter, delta,
                             batch_size=10, init_batch_size=20):

        def simulation(rules):
            rules = list(rules)
            if not rules:
                return []
            stats = lucb(self.evaluator, rules, beam_size, a=self.a,
                         max_iter=max_iter, delta=delta, batch_size=batch_size,
                         init_batch_size=init_batch_size)
            phis = [float(row[0]) for row in stats]
            self._record(rules, phis)
            return [(phi, phi) for phi in phis]
        return simulation

    def make_average_simulation(self, n_samples):
        def simulation(rules):
            rules = list(rules)
            if not rules:
                return []
            phis = [sum(self.sample_once(rule) for _ in range(n_samples)) / n_samples
                    for rule in rules]
            self._record(rules, phis)
            return [(phi, phi) for phi in phis]
        return simulation

    def isi_inputs(self, simulation):
        v = [self.actual_values[variable] for variable in self.variables]
        D = [list(self.domains[variable]) for variable in self.variables]
        dag = {variable: list(pa) for variable, pa in self.parents.items()}
        return (v, D, simulation, list(self.variables), dag,
                list(self.parents[self.target]))


# =========================================================================== #
#                                                                             #
#   brute-force ground truth and the metrics of one run                       #
#                                                                             #
# =========================================================================== #
from itertools import combinations  # noqa: E402  (already imported in block 1)

def ac2_pairs(scm) -> dict:
    """Every pair satisfying AC2, with the witnesses that realise it.

    Brute force: costs `(|Dom| + 1) ** |V|` evaluations, so it is meant for
    models of up to nine or ten variables.  This is the independent reference the
    algorithms are compared against -- it does not go through any of the search
    code.
    """
    variables = scm.variables
    actual = scm.actual_values
    pairs = {}
    for size in range(1, len(variables) + 1):
        for cause in combinations(variables, size):
            others = [variable for variable in variables if variable not in cause]
            alternatives = [[value for value in scm.domains[variable]
                             if value != actual[variable]] for variable in cause]
            for cause_values in product(*alternatives):
                intervention = dict(zip(cause, cause_values))
                witnesses = []
                for witness_size in range(len(others) + 1):
                    for witness in combinations(others, witness_size):
                        full = dict(intervention)
                        full.update({variable: actual[variable] for variable in witness})
                        if scm.cancels(full):
                            witnesses.append(frozenset(witness))
                if witnesses:
                    pairs[(frozenset(cause),
                           tuple(sorted(intervention.items())))] = witnesses
    return pairs

def actual_causes(scm, pairs=None, strict_ac3=True) -> list:
    """The actual causes of the target: the AC2 pairs that are minimal (AC3).

    `strict_ac3` selects the reading of AC3.  When set, a cause is a set no
    strict subset of which satisfies AC2 *for any values* -- the reading under
    which the minimal sets returned by an identification algorithm are exactly
    the causes.  The two readings coincide on Boolean models.
    """
    pairs = ac2_pairs(scm) if pairs is None else pairs
    ac2_sets = {cause for cause, _ in pairs}
    causes = []
    for cause, cause_values in pairs:
        if strict_ac3:
            minimal = not any(other < cause for other in ac2_sets)
        else:
            values = dict(cause_values)
            minimal = not any(
                (frozenset(subset),
                 tuple(sorted((variable, values[variable]) for variable in subset))) in pairs
                for size in range(1, len(cause))
                for subset in combinations(sorted(cause), size))
        if minimal:
            causes.append((cause, dict(cause_values)))
    return sorted(causes, key=lambda item: (len(item[0]), sorted(item[0])))

def dice_score(found_sets, reference_sets) -> float:
    """The paper's DICE, on sets of causes: 2 |A n B| / (|A| + |B|).

    Same definition as `evaluate_full`, restated here so that the ablation and
    the main results are measured with one formula.
    """
    total = len(found_sets) + len(reference_sets)
    if not total:
        return 1.0
    return 2 * len(found_sets & reference_sets) / total

def evaluate_ablation_run(solutions, true_causes, scm) -> dict:
    """Quality metrics of one run against the brute-force reference.

    `solved` is the paper's binary accuracy: the returned set of causes is
    exactly the reference set.  `ac2_validity` is the fraction of returned
    interventions that do cancel the target; it is the only metric that sees an
    unsound completion, and it is kept out of the table but reported for the
    `naive` assignment rows.
    """
    found_sets = {frozenset(solution[3]) for solution in solutions}
    reference_sets = {frozenset(cause) for cause in true_causes}
    recovered = found_sets & reference_sets
    valid = [scm.cancels(solution[0]) for solution in solutions]
    return {
        "n_true_causes": len(reference_sets),
        "max_cause_size": max((len(c) for c in reference_sets), default=0),
        "n_found": len(found_sets),
        "dice": dice_score(found_sets, reference_sets),
        "recall": len(recovered) / len(reference_sets) if reference_sets else 1.0,
        "precision": len(recovered) / len(found_sets) if found_sets else 1.0,
        "ac2_validity": sum(valid) / len(valid) if valid else 1.0,
        "solved": int(found_sets == reference_sets),
    }

# =========================================================================== #
#                                                                             #
#   corpus construction, configurations, the runner, the table and the figure #
#                                                                             #
# =========================================================================== #

ABLATION_EPSILON = 0.5
CORPUS_SIZES = (5, 6, 7, 8, 9, 10)
CORPUS_BIASES = (0.0, 0.3, 0.6, 0.85, 0.95)
CORPUS_DENSITIES = (0.35, 0.5, 0.65, 0.8)
CORPUS_DOMAINS = {5: (2, 3, 4), 6: (2, 3, 4), 7: (2, 3, 4),
                  8: (2, 3), 9: (2,), 10: (2,)}
CORPUS_DOMAINS_FALLBACK = (2,)

def domains_for_size(size: int, domain_sizes=CORPUS_DOMAINS) -> list:
    if isinstance(domain_sizes, dict):
        return list(domain_sizes.get(size, CORPUS_DOMAINS_FALLBACK))
    return list(domain_sizes)

def build_ablation_corpus(sizes=CORPUS_SIZES, n_per_size=30, domain_sizes=CORPUS_DOMAINS,
                          biases=CORPUS_BIASES, edge_probabilities=CORPUS_DENSITIES,
                          seed=0) -> list:
    """One corpus, drawn over a mixture of mechanism biases and densities.

    Models are stored as *specifications* -- the five arguments of
    `generate_random_scm` -- so the corpus is a few kilobytes and every model is
    regenerated deterministically.  One random stream per size, so that raising
    `n_per_size` extends the corpus instead of redrawing it.
    """
    corpus = []
    for size in tqdm(sizes, desc="Building corpus by model size"):
        rng = random.Random(seed)
        allowed_domains = domains_for_size(size, domain_sizes)
        kept = 0
        while kept < n_per_size:
            spec = {
                "n_variables": size,
                "domain_size": rng.choice(allowed_domains),
                "edge_probability": rng.choice(list(edge_probabilities)),
                "seed": rng.randrange(2 ** 31),
                "conjunction_bias": rng.choice(list(biases)),
            }
            scm = generate_random_scm(**spec)
            # models are redrawn until they have exactly the requested number of
            # variables, since irrelevant ones are pruned; models whose target
            # has no cause carry no information for an ablation
            if len(scm.variables) != size or not actual_causes(scm):
                continue
            spec["model_id"] = f"n{size}_{kept:04d}"
            corpus.append(spec)
            kept += 1
    return corpus

def load_ablation_model(spec: dict) -> RandomSCM:
    return generate_random_scm(spec["n_variables"], spec["domain_size"],
                               spec["edge_probability"], spec["seed"],
                               spec.get("conjunction_bias", 0.0))

def compute_ablation_truth(corpus: list) -> dict:
    return {spec["model_id"]: [sorted(cause) for cause, _
                               in actual_causes(load_ablation_model(spec))]
            for spec in tqdm(corpus, desc="Exact cause identification...")}

def show_corpus_statistic(corpus: list, truth: dict) -> dict:
    rows = []
    for spec in corpus:
        true_causes = truth[spec["model_id"]]
        rows.append({
            "model_id": spec["model_id"],
            "n_variables": spec["n_variables"],
            "domain_size": spec["domain_size"],
            "edge_probability": spec["edge_probability"],
            "conjunction_bias": spec.get("conjunction_bias", 0.0),
            "n_causes": len(true_causes),
            "max_cause_size": max((len(cause) for cause in true_causes), default=0),
        })
    frame = pd.DataFrame(rows)

    axes = ("n_variables", "domain_size", "max_cause_size")
    counts = {axis: frame[axis].value_counts().sort_index().rename("models")
              for axis in axes}
    joint = pd.crosstab(frame["n_variables"], frame["max_cause_size"])

    print(f"{len(frame)} models, {frame['n_causes'].sum()} causes in total")
    for axis in axes:
        print(f"\nmodels by {axis}")
        print(counts[axis].rename_axis(None).to_string())
    # print("\nn_variables (rows) x max_cause_size (columns)")
    # print(joint.to_string())

    # return {"per_model": frame,
    #         "n_variables_x_max_cause_size": joint,
    #         **counts}



# --------------------------------------------------------------------------- #
# configurations
# --------------------------------------------------------------------------- #
ABLATION_CONFIGURATIONS = [
    dict(name="ISI exact", family="exact", kind="isi", params=dict(
        exhaustive=True, backtrack="subsets", do_decomposition=True)),
    dict(name="MBS exact", family="exact", kind="mbs", params=dict(
        beam_size=-1, max_steps=-1, minimality=True)),

    dict(name="ISI exact, no decomposition", family="isi design", kind="isi", params=dict(
        exhaustive=True, backtrack="subsets", do_decomposition=False)),
    # dict(name="ISI exact, singleton backtrack", family="isi design", kind="isi", params=dict(
        # exhaustive=True, backtrack="singleton", do_decomposition=True)),

    *[dict(name=f"ISI exact, max_backtrack={depth}", family="isi budget", kind="isi",
           params=dict(exhaustive=True, max_backtrack=depth)) for depth in (1, 2, 3)],
    *[dict(name=f"ISI exact, sample={width}", family="isi budget", kind="isi",
           stochastic=True, params=dict(exhaustive=True, sample_backtrack=width))
      for width in (1, 4, 16, 64)],

    dict(name="ISI + MBS(-1,-1) minimal", family="isi+mbs", kind="isi", params=dict(
        exhaustive=False, minimal_only=True, beam_size=-1, max_steps=-1)),
    dict(name="ISI + MBS(-1,-1) all", family="isi+mbs", kind="isi", params=dict(
        exhaustive=False, minimal_only=False, beam_size=-1, max_steps=-1)),
    dict(name="ISI + MBS(8,5)", family="isi+mbs", kind="isi", params=dict(
        exhaustive=False, minimal_only=True, beam_size=8, max_steps=5)),
    dict(name="ISI + MBS(8,5), naive assign", family="isi+mbs", kind="isi", params=dict(
        exhaustive=False, minimal_only=True, beam_size=8, max_steps=5, assign="naive")),

    *[dict(name=f"MBS beam={size}", family="mbs beam", kind="mbs", params=dict(
        beam_size=size, max_steps=-1, minimality=True))
      for size in (4, 16, 256)],
    *[dict(name=f"MBS steps={steps}", family="mbs steps", kind="mbs", params=dict(
        beam_size=-1, max_steps=steps, minimality=True)) for steps in (2, 3, 5)],
    dict(name="MBS(8,5)", family="mbs beam", kind="mbs", params=dict(
        beam_size=8, max_steps=5, minimality=True)),

    dict(name="MBS, actual values in init", family="mbs mechanism", kind="mbs", params=dict(
        beam_size=-1, max_steps=-1, minimality=True, include_actual_initial=True)),
    dict(name="MBS, no superset pruning", family="mbs mechanism", kind="mbs", params=dict(
        beam_size=-1, max_steps=-1, minimality=True, prune_supersets=False)),
    dict(name="MBS, no pruning, expand causes", family="mbs mechanism", kind="mbs", params=dict(
        beam_size=-1, max_steps=-1, minimality=True,
        prune_supersets=False, expand_causes=True)),

    dict(name="ISI exact, +1 spurious parent", family="misspecified", kind="isi",
         params=dict(exhaustive=True), inflate=1),
    dict(name="ISI exact, +2 spurious parents", family="misspecified", kind="isi",
         params=dict(exhaustive=True), inflate=2),
    dict(name="ISI exact, -1 missing edge", family="misspecified", kind="isi",
         stochastic=True, params=dict(exhaustive=True), deflate=1),
    dict(name="ISI exact, -2 missing edges", family="misspecified", kind="isi",
         stochastic=True, params=dict(exhaustive=True), deflate=2),
]

def stochastic_configurations(noises=(0.02, 0.05, 0.10),
                              thresholds=(0.15, 0.30),
                              deltas=(0.05, 0.10),
                              budgets=(50, 200)) -> list:
    """LUCB against fixed-size averaging over the four parameters of the setting.

    Everything is run at one search setting, MBS with a beam of 8, so that the
    estimator and its parameters are the only things that vary.  The averaging
    baseline is only defined at the reference threshold, so LUCB and AVG must be
    compared at a = 0.30.
    """
    configurations = []
    for noise in noises:
        for budget in budgets:
            configurations.append(dict(
                name=f"AVG noise={noise}, N={budget}", family="stochastic avg",
                kind="stoch", stochastic=True,
                params=dict(beam_size=8, max_steps=5, minimality=True),
                stoch=dict(estimator="avg", noise=noise, a=0.30,
                           delta=0.10, budget=budget)))
            for delta in deltas:
                for threshold in thresholds:
                    configurations.append(dict(
                        name=f"LUCB noise={noise}, a={threshold}, "
                             f"delta={delta}, N={budget}",
                        family="stochastic lucb", kind="stoch", stochastic=True,
                        params=dict(beam_size=8, max_steps=5, minimality=True),
                        stoch=dict(estimator="lucb", noise=noise, a=threshold,
                                   delta=delta, budget=budget)))
    return configurations

# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def run_one_ablation(scm, true_causes, configuration, seed=0) -> dict:
    params = dict(configuration["params"])
    params.setdefault("epsilon", ABLATION_EPSILON)
    random.seed(seed)
    np.random.seed(seed)

    noisy = None
    if configuration["kind"] == "stoch":
        stoch = configuration["stoch"]
        noisy = NoisyRandomSCM(scm, stoch["noise"], stoch["a"], seed=seed)
        if stoch["estimator"] == "lucb":
            simulation = noisy.make_lucb_simulation(
                beam_size=params.get("beam_size", 8),
                max_iter=stoch["budget"], delta=stoch["delta"])
        else:
            simulation = noisy.make_average_simulation(stoch["budget"])
        params["epsilon"] = stoch["a"]
        v, D, _, V, dag, PA_T = noisy.isi_inputs(simulation)
        noisy.reset_counter()
    else:
        v, D, simulation, V, dag, PA_T = scm.isi_inputs()
        if configuration.get("inflate"):
            dag, PA_T = inflate_parents(scm, configuration["inflate"], seed)
        if configuration.get("deflate"):
            dag, PA_T = deflate_parents(scm, configuration["deflate"], seed)
        scm.reset_counter()

    solver, timed_out = None, False
    start = time.time()
    if configuration["kind"] == "isi":
        solutions = iterative_identification(v, D, simulation, V, dag, PA_T, target=scm.target, **params)
    else:
        solutions = beam_search(v=v, D=D, simulation=simulation, V=V, **params)
        if isinstance(solutions, tuple):
            solutions = solutions[0]
    seconds = time.time() - start

    if noisy is not None:
        n_calls = noisy.n_calls
        wrong = [estimated != true for estimated, true in noisy.decisions]
        decision_error = sum(wrong) / len(wrong) if wrong else 0.0
    else:
        n_calls, decision_error = scm.n_calls, None

    row = {
        "configuration": configuration["name"],
        "family": configuration["family"],
        "model_id": scm.model_id,
        "n_variables": len(scm.variables),
        "seed": seed,
        "n_calls": n_calls,
        "n_subproblems": len(solver.seen) if solver is not None else None,
        "seconds": seconds,
        "decision_error": decision_error,
        "timed_out": int(timed_out),
    }
    row.update(evaluate_ablation_run(solutions, true_causes, scm))
    return row

def _format_ablation_row(row, columns):
    return ",".join(
        "" if row[column] is None else
        (f'"{row[column]}"' if isinstance(row[column], str) else str(row[column]))
        for column in columns)

def run_ablation_study(corpus, truth, configurations, n_seeds=5,
                       verbose=True, path=None) -> list:
    """Run every configuration on every model.

    When `path` is given, each model's rows are appended as soon as they are
    produced and models already in the file are skipped, so an interrupted study
    is relaunched by calling this again with the same arguments.
    """
    done, columns = set(), None
    if path and os.path.exists(path):
        # configuration names contain commas, so the file must be read as CSV
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            for record in reader:
                done.add(record["model_id"])
        if verbose:
            print(f"  resuming: {len(done)} models already done", flush=True)

    rows = []
    for index, spec in enumerate(tqdm(corpus,desc="Enumerate corpus...",leave=False)):
        if spec["model_id"] in done:
            continue
        scm = load_ablation_model(spec)
        scm.model_id = spec["model_id"]
        true_causes = truth[spec["model_id"]]
        model_rows = []
        for configuration in configurations:
            seeds = range(n_seeds) if configuration.get("stochastic") else [0]
            for seed in seeds:
                model_rows.append(
                    run_one_ablation(scm, true_causes, configuration, seed))
        rows += model_rows
        if path:
            if columns is None:
                columns = list(model_rows[0].keys())
                with open(path, "w") as handle:
                    handle.write(",".join(columns) + "\n")
            with open(path, "a") as handle:
                for row in model_rows:
                    handle.write(_format_ablation_row(row, columns) + "\n")
        if verbose:
            print(f"  {index + 1}/{len(corpus)} models ({spec['model_id']})", flush=True)
    return rows

# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
ABLATION_FAMILY_BASELINE = {
    "exact": None, "isi design": "ISI exact", "isi budget": "ISI exact",
    "isi+mbs": "ISI exact", "mbs beam": "MBS exact", "mbs steps": "MBS exact",
    "mbs mechanism": "MBS exact", "misspecified": "ISI exact",
}
ABLATION_FAMILY_ORDER = ["exact", "isi design", "isi budget", "isi+mbs",
                         "mbs beam", "mbs steps", "mbs mechanism", "misspecified"]

def summarise_ablation(df):
    """One row per configuration: means for quality, medians for cost.

    The call ratio is a *per-model* ratio against the family's reference
    configuration, aggregated afterwards -- not the ratio of the medians.
    """
    import pandas as pd

    per_model = (df.groupby(["configuration", "family", "model_id"])
                   .agg(solved=("solved", "mean"), dice=("dice", "mean"),
                        n_calls=("n_calls", "mean"),
                        n_subproblems=("n_subproblems", "mean"))
                   .reset_index())
    calls = per_model.pivot_table(index="model_id", columns="configuration",
                                  values="n_calls")
    ratios = {}
    for configuration, family in per_model[["configuration", "family"]].drop_duplicates().values:
        baseline = ABLATION_FAMILY_BASELINE.get(family)
        if baseline is None or baseline not in calls.columns:
            ratios[configuration] = np.nan
        else:
            ratios[configuration] = (calls[configuration]
                                     / calls[baseline].replace(0, np.nan)).median()

    summary = (per_model.groupby(["configuration", "family"])
                        .agg(solved=("solved", "mean"), dice=("dice", "mean"),
                             calls=("n_calls", "median"),
                             subproblems=("n_subproblems", "median"))
                        .reset_index())
    summary["call_ratio"] = summary["configuration"].map(ratios)
    listed = {configuration["name"] for configuration in ABLATION_CONFIGURATIONS}
    summary = summary[summary["configuration"].isin(listed)]
    summary["family"] = pd.Categorical(summary["family"], ABLATION_FAMILY_ORDER,
                                       ordered=True)
    order = {c["name"]: i for i, c in enumerate(ABLATION_CONFIGURATIONS)}
    summary["_order"] = summary["configuration"].map(order)
    return summary.sort_values(["family", "_order"]).drop(columns="_order")

def latex_ablation_table(summary, path) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Configuration & Family & Solved & DICE & Calls & Ratio \\",
             r"\midrule"]
    previous = None
    for _, row in summary.iterrows():
        if previous is not None and row["family"] != previous:
            lines.append(r"\addlinespace")
        previous = row["family"]
        ratio = "--" if pd.isna(row["call_ratio"]) else f"{row['call_ratio']:.2f}"
        lines.append(f"{row['configuration'].replace('_', chr(92) + '_')} & "
                     f"{row['family']} & {row['solved']:.2f} & {row['dice']:.2f} & "
                     f"{row['calls']:.0f} & {ratio} \\\\")
        
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Ablation on the mixed random corpus. Solved and DICE are means over models, costs are medians. \emph{Ratio} is the median per-model number of calls relative to the reference configuration of the same family (\emph{ISI exact} or \emph{MBS exact}).}", r"\label{tab:ablation}", r"\end{table}"]
    text = "\n".join(lines)
    with open(path, "w") as handle:
        handle.write(text + "\n")
    return text

def _annotate_stochastic(df):
    df = df.copy()
    df["estimator"] = df["configuration"].str.split(" ").str[0]
    df["noise"] = df["configuration"].str.extract(r"noise=([0-9.]+)").astype(float)
    df["a"] = df["configuration"].str.extract(r"a=([0-9.]+)").astype(float).fillna(0.30)
    df["delta"] = df["configuration"].str.extract(r"delta=([0-9.]+)").astype(float).fillna(0.10)
    df["budget"] = df["configuration"].str.extract(r"N=(\d+)").astype(int)
    return df

def summarise_stochastic(df, reference_threshold=0.30, reference_delta=0.10,
                         grid_budget=200):
    """LUCB against averaging, and LUCB's sensitivity to its own parameters.

    Both frames carry a `ratio` column: the number of calls relative to the
    averaging baseline at the same noise level and budget.  The averaging
    baseline is only defined at the reference threshold, so the comparison in
    `matched` is made there; `grid` varies the threshold and the confidence level
    at a fixed budget.
    """
    df = _annotate_stochastic(df)
    columns = ["solved", "dice", "n_calls", "decision_error"]
    baseline = (df[df["estimator"] == "AVG"]
                .groupby(["noise", "budget"])["n_calls"].mean())

    keep = ((df["estimator"] == "AVG")
            | ((df["a"] == reference_threshold) & (df["delta"] == reference_delta)))
    matched = (df[keep].groupby(["noise", "budget", "estimator"])[columns]
                       .mean().reset_index())
    matched["ratio"] = [row.n_calls / baseline.loc[(row.noise, row.budget)]
                        for row in matched.itertuples()]

    grid = (df[(df["estimator"] == "LUCB") & (df["budget"] == grid_budget)]
            .groupby(["noise", "a", "delta"])[columns].mean().reset_index())
    grid["ratio"] = [row.n_calls / baseline.loc[(row.noise, grid_budget)]
                     for row in grid.itertuples()]
    return matched, grid

def latex_stochastic_table(matched, grid, path) -> str:
    def calls(value):
        return f"{value / 1000:.1f}" if value >= 1000 else f"{value:.0f}"

    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{subtable}{\linewidth}", r"\centering",
             r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"$\epsilon_n$ & Estimator & Solved & DICE & Calls ($10^3$) "
             r"& Ratio & Err. \\", r"\midrule"]
    previous = None
    for _, row in matched.sort_values(["noise", "budget", "estimator"]).iterrows():
        if previous is not None and row["noise"] != previous:
            lines.append(r"\addlinespace")
        previous = row["noise"]
        name = f"{row['estimator']}, $N={int(row['budget'])}$"
        lines.append(f"{row['noise']:.2f} & {name} & {row['solved']:.2f} & "
                     f"{row['dice']:.2f} & {calls(row['n_calls'])} & "
                     f"{row['ratio']:.2f} & {row['decision_error']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Adaptive sampling against fixed-size averaging, "
              r"at the reference threshold $a=0.30$ and $\delta=0.10$.}",
              r"\label{tab:lucb-matched}", r"\end{subtable}",
              r"", r"\vspace{1em}", r"",
              r"\begin{subtable}{\linewidth}", r"\centering",
              r"\begin{tabular}{lllrrrrr}", r"\toprule",
              r"$\epsilon_n$ & $a$ & $\delta$ & Solved & DICE & "
              r"Calls ($10^3$) & Ratio & Err. \\", r"\midrule"]
    previous = None
    for _, row in grid.sort_values(["noise", "a", "delta"]).iterrows():
        if previous is not None and row["noise"] != previous:
            lines.append(r"\addlinespace")
        previous = row["noise"]
        lines.append(f"{row['noise']:.2f} & {row['a']:.2f} & {row['delta']:.2f} & "
                     f"{row['solved']:.2f} & {row['dice']:.2f} & "
                     f"{calls(row['n_calls'])} & {row['ratio']:.2f} & "
                     f"{row['decision_error']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Sensitivity of LUCB to the canceling threshold and the "
              r"confidence level, at $N=200$.}",
              r"\label{tab:lucb-grid}", r"\end{subtable}",
              r"\caption{...}", r"\label{tab:stochastic}", r"\end{table}"]
    text = "\n".join(lines)
    with open(path, "w") as handle:
        handle.write(text + "\n")
    return text

# --------------------------------------------------------------------------- #
# the cause-size figure
# --------------------------------------------------------------------------- #
def cause_size_frame(df, cap: int = 4):
    """Per-model ISI / MBS comparison, binned by measured maximum cause size."""
    exact = df[df["configuration"].isin(["ISI exact", "MBS exact"])]
    wide = exact.pivot_table(index=["model_id", "n_variables", "max_cause_size"],
                             columns="configuration",
                             values=["n_calls", "dice", "solved"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns.values]
    wide["bin"] = wide["max_cause_size"].clip(upper=cap)
    wide["log_ratio"] = np.log2(wide["n_calls_ISI exact"] / wide["n_calls_MBS exact"])
    return wide

def plot_cause_size(wide, path, cap: int = 4):
    """Two panels: absolute cost, then the ratio broken down by graph size.

    The second panel exists to answer the obvious objection to the first: larger
    causes come from larger graphs in this corpus, so the ratio is shown within
    each |V| to establish that the effect is not a graph-size artefact.
    """
    import matplotlib.pyplot as plt

    bins = sorted(wide["bin"].unique())
    labels = [f"{int(b)}" if b < cap else f"$\\geq{cap}$" for b in bins]
    counts = [int((wide["bin"] == b).sum()) for b in bins]

    fig, axes = plt.subplots(2, 1, figsize=(5.4, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1]})
    for name, colour, marker, label in (("ISI exact", "tab:blue", "o", "ISI"),
                                        ("MBS exact", "tab:orange", "s", "MBS")):
        median = [wide[wide["bin"] == b][f"n_calls_{name}"].median() for b in bins]
        low = [wide[wide["bin"] == b][f"n_calls_{name}"].quantile(.25) for b in bins]
        high = [wide[wide["bin"] == b][f"n_calls_{name}"].quantile(.75) for b in bins]
        axes[0].plot(range(len(bins)), median, marker=marker, color=colour, label=label)
        axes[0].fill_between(range(len(bins)), low, high, color=colour, alpha=.15)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("model calls")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.3)

    sizes = sorted(wide["n_variables"].unique())
    cmap = plt.get_cmap("viridis")
    for index, size in enumerate(sizes):
        sub = wide[wide["n_variables"] == size]
        xs, ys = [], []
        for position, b in enumerate(bins):
            values = sub[sub["bin"] == b]["log_ratio"].dropna()
            if len(values) >= 3:                  # do not draw a point on one model
                xs.append(position)
                ys.append(values.median())
        if len(xs) >= 2:
            axes[1].plot(xs, ys, marker="o", ms=4, lw=1.4,
                         color=cmap(index / max(1, len(sizes) - 1)),
                         label=f"$|V|={size}$")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel(r"$\log_2$(ISI / MBS calls)")
    axes[1].legend(frameon=False, fontsize=7, ncol=3, loc="lower left")
    axes[1].grid(alpha=.3)
    axes[1].set_xticks(range(len(bins)))
    axes[1].set_xticklabels([f"{l}\n(n={c})" for l, c in zip(labels, counts)])
    axes[1].set_xlabel("maximum cause size")
    fig.tight_layout()
    fig.savefig(path)
    return fig

if __name__ == "__main__":
    # Create the corpus and find the solutions
    corpus = build_ablation_corpus(n_per_size=50)
    truth  = compute_ablation_truth(corpus)
    show_corpus_statistic(corpus, truth)

    # Make the ablation study table
    # run_ablation_study(corpus, truth, ABLATION_CONFIGURATIONS,
    #                    path="results_ablation/ablation.csv", verbose=False)
    df = pd.read_csv("results_ablation/ablation.csv")
    sumary = summarise_ablation(df)
    latex_ablation_table(sumary, path="results_ablation/table-ablation.tex")

    # Make the cause size figure
    wide = cause_size_frame(df)
    plot_cause_size(wide, path="results_ablation/fig-ablation.pdf")
    
    # Make the stochastic study table
    run_ablation_study(corpus, truth, stochastic_configurations(),
                       n_seeds=3, path="results_ablation/ablation_stochastic.csv", verbose=False)
    df_stoc = pd.read_csv("results_ablation/ablation_stochastic.csv")
    matched, grid = summarise_stochastic(df_stoc)
    latex_stochastic_table(matched, grid, path="results_ablation/table-ablation_stoc.tex")