import os
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

from dag_generator import *
import json
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import copy

# From repo (to remove)
def evaluate_full(causes, ref_causes):
    n_minimal = 0
    n_non_minimal = 0
    n_missed = 0
    avg_overshot = 0
    for ref_cause in ref_causes:
        for cause in causes:
            if set(ref_cause) < set(cause):
                n_non_minimal += 1
                avg_overshot += len(cause) - len(ref_cause)
                break
            if set(ref_cause) == set(cause):
                n_minimal += 1
                break
        else:
            n_missed += 1
    
    p = n_minimal / len(causes) if len(causes) else 0
    r = n_minimal / len(ref_causes) if len(ref_causes) else 1
    
    target = {tuple(sorted(cause)) for cause in ref_causes}
    pred = {tuple(sorted(cause)) for cause in causes}
    
    return {
        "Accuracy": int(set(causes) == set(ref_causes)),
        "Recall": r,
        "Precision": p,
        "jaccard": len(target & pred) / len(target | pred) if len(target | pred) else 1,
        "dice": 2 * len(target & pred) / (len(target) + len(pred)) if len(target) + len(pred) else 1,
        "F1": 2 * p * r / (p + r) if p+r else 0, 
        "Missed": n_missed / len(ref_causes) if ref_causes else 1,
        "% Overshoot": n_non_minimal / len(ref_causes) if ref_causes else 1,
        "Average Overshoot": avg_overshot / n_non_minimal if n_non_minimal else 0,
        "Average Repeat": (
            (len(causes) - n_minimal) / n_non_minimal if 
            n_non_minimal 
            else 0
            )
     }

# Code to move
@dataclass
class IdentificationParam:
    ISI: bool
    beam_size: int
    max_steps: int

@dataclass
class MBSParam(IdentificationParam):
    pass

@dataclass
class ISIParam(IdentificationParam):
    minimal_only: bool
    assign: str


n = 20
n_models = 15
theta = 0.5
max_steps = 10
growth_threshold = 0.01

beam_sizes = (1,2,4,8,16,32,64,128,256)

experiments_table_1 = {
    "layer":[
        (LayerParams(L=5, w=3, p=.5), "base"),
        (LayerParams(L=6, w=3, p=.5), "long"),
        (LayerParams(L=3, w=5, p=.5), "large"),
        (LayerParams(L=5, w=3, p=.7), "dense")
    ],
    "shortcut":[
        (TreeParams(n=15, r=5), "base"),
        (TreeParams(n=15, r=7), "dense"),
        (TreeParams(n=20, r=5), "large")
    ],
    "fanin":[
        (FaninParams(n=20, k=3, k_init=5), "base"),
        (FaninParams(n=20, k=5, k_init=5), "dense"),
        (FaninParams(n=30, k=4, k_init=5), "large")
    ],
    "er":[
        (ERParams(n=15, avg_degree=2, k_init=5), "base"),
        (ERParams(n=15, avg_degree=3, k_init=5), "dense"),
        (ERParams(n=20, avg_degree=2, k_init=5), "large")
    ],
    # "bottleneck":[
    #     (BottleneckParams(d=7, w=7), "base"),
    #     (BottleneckParams(d=9, w=5), "long"),
    #     (BottleneckParams(d=4, w=8), "large")
    # ]
}

identification_params = {
    "ISI-small": ISIParam(ISI=True , beam_size=2,   max_steps=10, assign="naive", minimal_only=True),
    "ISI-large": ISIParam(ISI=True , beam_size=32, max_steps=10, assign="naive", minimal_only=True),
    "MBS-small": MBSParam(ISI=False, beam_size=2,   max_steps=10),
    "MBS-large": MBSParam(ISI=False, beam_size=32, max_steps=10),
}

SWEEP_LAYER = {
    "layer-density": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "layer-theta":   [0.4, 0.5, 0.6, -0.4, -0.5, -0.6, "parity"],
    "layer-depth":   [2, 3, 4, 5, 6, 7, 8],
    "layer-width":   [2, 3, 4, 5, 6, 7],
}

# # Small test sweep
# SWEEP_LAYERS_SMALL = {
#     "layer-width":   [2, 3],
#     "layer-depth":   [2, 3, 4],
#     "layer-density": [0.3, 0.4, 0.5],
#     "layer-theta":   [0.4, 0.5, 0.6, -0.4, -0.5, -0.6, "parity"],
# }

SWEEP_TREES = {
    "shortcut-n":     [10,15,20,25,30,35,40],
    "shortcut-r":     [0,2,4,6,8,10,12],
    "shortcut-theta": [0.4, 0.5, 0.6, -0.4, -0.5, -0.6, "parity"],
    "shortcut-bias":  [-3, -2, -1, 0, 1, 2, 3],
}

# SWEEP_TREES_SMALL = {
#     "shortcut-n": [10,15,20],
#     "shortcut-r": [1,2,3,4],
#     "fanin-n":    [10,15,20],
#     "fanin-k":    [1,2,3],
# }

# Utils
def half_ones_permutations(n: int) -> np.ndarray:
    """
    Returns a boolean numpy array of shape (C(n, n//2), n) where each row
    is a unique binary vector with exactly n//2 ones.
    If n is odd, returns all rows with either floor(n/2) or ceil(n/2) ones,
    i.e., all possible balanced-or-near-balanced permutations.
    """
    half = n // 2
    counts = [half, n - half] if n % 2 != 0 else [half]
    
    rows = [
        np.array([1 if i in ones_idx else 0 for i in range(n)], dtype=int)
        for k in counts
        for ones_idx in combinations(range(n), k)
    ]
    return np.array(rows)

def get_us(model, is_bottleneck=False):
    
    if is_bottleneck:
        us = np.random.randint(0,2, size=(200, len(model.U)))
        us[:, 0] = 1
    else:
        us = half_ones_permutations(len(model.U))
    return us

def get_exact_solutions(model, us, verbose=False):
    scms = []
    for u in tqdm(us, disable=not verbose, desc="Finding solutions", leave=False):
        scm = build_scm(model.dag, u, model.F)
        # scm.find_causes(ISI=True, exhaustive=True)
        scm.find_causes(ISI=False, beam_size=-1, max_steps=-1)
        scms.append(scm)
    return scms

def make_json(model, u_list, scms, dices=None):
    json_data = {
        "model": model.to_json(),
        "us": u_list if isinstance(u_list, list) else u_list.tolist(),
        "causes": [scm.causes_hashable for scm in scms],
        "witnesses": [scm.witnesses_hashable for scm in scms],
        "identification_times": [scm.identification_time for scm in scms],
        "n_calls": [scm.n_calls for scm in scms],
    } 
    if dices is not None:
        json_data["dices"] = dices
    return json_data

def build_models(dag_fnt_label, params, rng, theta=.5, n=50, n_models=15):
    dag_fnt = dag_fnts[dag_fnt_label]
    is_bottleneck = dag_fnt_label == "bottleneck"

    models = []
    for _ in range(n_models):
        if is_bottleneck:
            dag, F = dag_fnt(**asdict(params))
        else:
            dag = dag_fnt(**asdict(params), rng=rng)
            F = make_F(dag, theta=theta)
        models.append(build_model(dag, F))
    pos = []
    for i, model in enumerate(models):
        us = get_us(model, is_bottleneck)
        for u in us:
            if model(u)[-1]:
                pos.append((i,u.tolist()))
    choice = np.random.choice(len(pos), size=n, replace=False)

    # Group selected us by model index
    grouped = defaultdict(list)
    for i in choice:
        mi, u = pos[i]
        grouped[mi].append(u)
    return [models[mi] for mi in grouped.keys()], [grouped[mi] for mi in grouped.keys()]

def compute_density(dag):
    return np.mean(list(map(len, dag.values())))

def get_sweeped_params(model, sweep_label, sweep_value, theta, params):
    theta_val = theta
    params = copy.deepcopy(params)
    if model == "layer":
        if sweep_label == "width":
            params.w = sweep_value
        elif sweep_label == "depth":
            params.L = sweep_value
        elif sweep_label == "density":
            params.p = sweep_value
    elif model == "shortcut":
        if sweep_label == "n":
            params.n = sweep_value
        elif sweep_label == "r":
            params.r = sweep_value
    elif model == "fanin":
        if sweep_label == "n":
            params.n = sweep_value
        elif sweep_label == "k":
            params.k = sweep_value
    if sweep_label == "theta":
        theta_val = sweep_value
    return params, theta_val

# Prepare models and exact solutions
def make_table1_data(experiments, rng, theta, n, n_models):
    for model_label, param_list in experiments.items():
        os.makedirs(f"results_structure/table1/{model_label}", exist_ok=True)
        for params, param_label in param_list:
            if os.path.exists(f"results_structure/table1/{model_label}/{param_label}.json"):
                print(f"Skipping {model_label}-{param_label} as it already exists.")
                continue
            res = []
            print(f"Testing {model_label} with params {params}")
            models, us = build_models(model_label, params, rng, theta, n=n, n_models=n_models)
            for model, u_list in tqdm(list(zip(models, us)), desc=f"Testing {param_label}"):
                scms = get_exact_solutions(model, u_list)
                res.append(make_json(model, u_list, scms))
            json_data = {
                "params": asdict(params),
                "data": res
            }
            with open(f"results_structure/table1/{model_label}/{param_label}.json", "w") as f:
                json.dump(json_data, f, indent=4)

def make_sensitivity_data(sweeps, rng, base_theta, base_params, n, n_models):
    for sweep_label, sweep_values in tqdm(sweeps.items(), desc="Sweep values", leave=True):
        if os.path.exists(f"results_structure/sensitivity/{sweep_label}.json"):
            print(f"Skipping {sweep_label} as it already exists.")
            continue
        os.makedirs("results_structure/sensitivity/", exist_ok=True)
        model_label, sweep_label = sweep_label.split("-")
        base_params_model = base_params[model_label]
        json_data = {
            "sweep_label": sweep_label,
            "model": model_label,
            "sweep_values": sweep_values,
            "base_params": asdict(base_params_model),
            "base_theta": base_theta,
            "data": {}
        }
        for sweep_value in tqdm(sweep_values, desc=f"Sweeping through {sweep_label}"):
            params, theta_val = get_sweeped_params(model_label, sweep_label, sweep_value, base_theta, base_params_model)

            models, us = build_models(model_label, params, rng, theta=theta_val, n=n, n_models=n_models)
            res = []
            for model, u_list in tqdm(list(zip(models, us)), desc=f"Browsing models for {sweep_label}={sweep_value}", leave=False):
                scms = get_exact_solutions(model, u_list, verbose=True)
                res.append(make_json(model, u_list, scms))
            json_data["data"][sweep_value] = res
        with open(f"results_structure/sensitivity/{'-'.join([model_label, sweep_label])}.json", "w") as f:
                json.dump(json_data, f, indent=4)

# Compute evaluation based on results of approximate approaches and exact solutions
def compute_growth_values(dices_array, calls_array, threshold = .01, verbose=0):
    dices = np.median(dices_array, axis=1)
    calls = np.median(calls_array, axis=1)

    # ratio per consecutive beam-size step: Δdice / Δlog2(calls)
    log2_calls = np.log2(np.maximum(calls, 1))  # guard against 0
    delta_dice  = np.diff(dices)                 # shape (7,)
    delta_log2c = np.diff(log2_calls)                # shape (7,)

    valid = delta_log2c > 0.001
    ratios = np.where(valid, delta_dice / delta_log2c, 0)
    growing    = int(np.sum(ratios >  threshold))
    flat       = int(np.sum((ratios >= 0) & (ratios <= threshold)))
    decreasing = int(np.sum(ratios < 0))
    if verbose:
        print(f"min_dice={dices.min():.3f}  max_dice={dices.max():.3f}  "
            f"min_calls={calls.min():.0f}  max_calls={calls.max():.0f}  "
            f"{growing=}  {flat=}  {decreasing=}")
    return dices.min(), dices.max(), calls.min(), calls.max(), growing, flat, decreasing

# Execute approximate approaches
def evaluate_model_params(model_label, param_label, do_isi, threshold, verbose=0):
    try:
        with open(f"results_structure/table1/{model_label}/{param_label}.json", "r") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        return
    
    dices_array = []
    calls_array = []
    for bs in tqdm(beam_sizes, disable=verbose < 2):
        dices = []
        calls = []
        for datum in json_data["data"]:
            for u, causes in zip(datum["us"], datum["causes"]):
                scm = build_scm(**datum["model"], u=u)
                args = {}
                if do_isi:
                    args["minimal_only"] = True
                    args["assign"] = "naive"
                scm.find_causes(ISI=do_isi, beam_size=bs, max_steps=max_steps, **args)

                ev = evaluate_full(scm.causes_hashable, [tuple(cause) for cause in causes])
                dices.append(ev["dice"])
                calls.append(scm.n_calls)
        dices_array.append(dices)
        calls_array.append(calls)
    return compute_growth_values(dices_array, calls_array, threshold = threshold, verbose=verbose)

def compute_sweep_results(sweep_label, id_label):
    # Load reference
    try:
        with open(f"results_structure/sensitivity/{sweep_label}.json", "r") as f:
            json_data_ref = json.load(f)
    except FileNotFoundError:
        return
    
    # Copy model params
    json_data = {key:json_data_ref[key] for key in ("sweep_label", "sweep_values", "base_params", "base_theta")}

    # Retrieve identification params
    id_params = identification_params[id_label]

    # Execute approximate searches
    json_data["data"] = {}
    for sweep_value, data in tqdm(json_data_ref["data"].items(), leave=False, desc=f"Params {id_label}"):
        res = []
        for datum in tqdm(data, desc=f"Models for {sweep_label}={sweep_value}", leave=False):
            dices = []
            scms = []
            for u, ref_causes in zip(datum["us"], datum["causes"]):
                scm = build_scm(**datum["model"], u=u)
                scm.find_causes(**asdict(id_params))
            
                ev = evaluate_full(scm.causes_hashable, [tuple(cause) for cause in ref_causes])
                dices.append(ev["dice"])
                scms.append(scm)
            res.append(make_json(build_model(**datum["model"]), datum["us"], scms, dices))
        json_data["data"][sweep_value] = res
    
    os.makedirs(f"results_structure/sensitivity/{id_label}", exist_ok=True)
    with open(f"results_structure/sensitivity/{id_label}/{sweep_label}.json", "w") as file:
        file.write(json.dumps(json_data, indent=2))

def retrieve_sweep_scores(id_label, sweep_label, sweeps):
    # Load results
    if id_label=="exact":
        path = f"results_structure/sensitivity/{sweep_label}.json"
    else:
        path = f"results_structure/sensitivity/{id_label}/{sweep_label}.json"
    with open(path, "r") as f:
        json_data = json.load(f)

    dice_array = []
    call_array = []
    # Show results
    assert json_data["sweep_values"] == sweeps[sweep_label]
    for sweep_value in json_data["sweep_values"]:
        dice_values = []
        call_values = []
        for datum in json_data["data"][str(sweep_value)]:
            if not id_label=="exact": 
                dice_values.extend(datum["dices"])
            else:
                dice_values.append([1] * len(datum["us"]))
            call_values.extend(datum["n_calls"])
        dice_array.append(dice_values)
        call_array.append(call_values)
    return dice_array, call_array

# Make plots and tables
# Plot for sweeps
def plot_sweep_value(i, id_label, sweep, sweep_label, metric, ax, width = 0.19):
    print(i, id_label, sweep_label, metric)
    arr = retrieve_sweep_scores(id_label, sweep_label, sweep)[int(metric == "n_calls")]
    n = len(sweep[sweep_label]) + (metric == "n_calls")
    if "MBS" in id_label:
        ls = "-"
    elif "ISI" in id_label:
        ls = "--"
    else:
        ls = "-."
    x = np.arange(len(sweep[sweep_label]))
    # y = np.mean(arr, axis=1)
    y = np.median(arr, axis=1)
    # y_err = np.std(arr, axis=1)
    y_min = np.quantile(arr, .25, axis=1)
    y_max = np.quantile(arr, .75, axis=1)
    y_err = [y-y_min, y_max-y]
    if "theta" in sweep_label:
        offset = width * (i - (n - 1) / 2)   # centers the group on the tick
        ax.bar(x + offset, y, width=width, yerr=y_err, ls=ls)
    else:
        ax.errorbar(x, y, yerr=y_err, linestyle=ls)

def plot_one_sweep(sweep, sweep_label, metric, ax=None, show=False):
    if ax is None:
        ax = plt.gca()
    # y_offset = 5 if metric == "n_calls" else .5
    for i, id_label in enumerate(("MBS-small", "MBS-large", "ISI-small", "ISI-large")):
        plot_sweep_value(i, id_label, sweep, sweep_label, metric, ax)
    if metric == "n_calls":
        plot_sweep_value(i+1, "exact", sweep, sweep_label, metric, ax)
    ax.set_xticks(np.arange(len(sweep[sweep_label])), list(map(str,sweep[sweep_label])), rotation=45)
    if show:
        ax.legend()
        plt.show()

def plot_legend_sweep(sweep, ax, loc):
    for i, id_label in enumerate(("MBS-small", "MBS-large", "ISI-small", "ISI-large")):
        ls = "-" if "MBS" in id_label else "--"
        ax.plot([], [], label=id_label, ls=ls, color=f"C{i}")
    ax.plot([], [], label="exact", ls="-.", color=f"C{i+1}")
    ax.legend(loc=loc)
    # ax.set_axis_off()
    # ax.set_axis_off()

def plot_sweep(sweep, path):
    w,h=3.1,2.1
    figsize=(w*4,h*2)
    _, axes = plt.subplots(2, 4, figsize=figsize, sharex="col", sharey="row")
    for metric, ax_line in zip(("dice", "n_calls"), axes):
        for sweep_label, ax in zip(sweep, ax_line):
            plot_one_sweep(sweep, sweep_label, metric, ax)
    axes[0, 0].set_ylabel("dice")
    axes[1, 0].set_ylabel("n_calls")
    for i, sweep_label in enumerate(sweep):
        axes[1, i].set_xlabel(sweep_label.split("-")[-1])
        axes[1, i].set_yscale("log")
    plot_legend_sweep(sweep, axes[0, 0], loc="lower left")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

# Table for generalization across model structures
def build_latex_table(experiments, threshold):
    rows = []
    for model_label, param_list in tqdm(experiments.items(), desc="Models", leave=True):
        for _, param_label in tqdm(param_list, desc=model_label):
            for do_isi in (False, True):

                min_d, max_d, min_c, max_c, growing, flat, decreasing = \
                    evaluate_model_params(model_label, param_label, do_isi, threshold, verbose=0)

                rows.append({
                    "Graph":     f"{model_label}-{param_label}",
                    "Algo":      "ISI" if do_isi else "MBS",
                    "Min DICE":  f"{min_d:.2f}",
                    "Max DICE":  f"{max_d:.2f}",
                    "Min calls": f"{min_c:.0f}",
                    "Max calls": f"{max_c:.0f}",
                    "G / F / D": f"{growing} / {flat} / {decreasing}",
                })
    df = pd.DataFrame(rows).set_index(["Graph", "Algo"])

    latex = df.to_latex(
        multirow=True,
        multicolumn=True,
        escape=True,
        column_format="ll" + "r" * len(df.columns),
    )

    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        + latex +
        f"\\caption{{DICE score and number of oracle calls across graph types and algorithms, "
        f"for beam sizes $b \\in \\{{2, 4, 8, 16, 32, 64, 128, 256\\}}$. "
        f"G\\,/\\,F\\,/\\,D counts the number of beam-size steps (out of 7) "
        f"where the DICE-per-log-call ratio is growing, flat, or decreasing "
        f"(threshold $= {threshold}$).}}\n"
        f"\\label{{tab:dag_types}}\n"
        "\\end{table}\n"
    )

    return latex, df


if __name__ == "__main__":
    rng = np.random.default_rng(42)


    # # Table 1: do the “beam-size vs nb calls” generalizes to other model structures?
    make_table1_data(experiments_table_1, rng, theta, n, n_models)
    latex, df = build_latex_table(experiments_table_1, growth_threshold)
    print(df.to_string())
    with open("results_structure/table1/table1.tex", "w") as f:
        f.write(latex)

    # Do structural properties influence the capabilities of MBS and ISI?
    BASE_LAYER_PARAMS = LayerParams(L=3, w=3, p=.4)   # base for layer-* columns
    BASE_TREE_PARAMS  = TreeParams(n=15, r=3, bias=0) # base for shortcut-* columns
    # BASE_FANIN_PARAMS = FaninParams(n=20, k=3, k_init=5)  # base for fanin-* columns
    base_params = {
        "layer": BASE_LAYER_PARAMS,
        "shortcut": BASE_TREE_PARAMS,
    }

    make_sensitivity_data(SWEEP_LAYER, rng, theta, base_params, n, n_models)
    make_sensitivity_data(SWEEP_TREES, rng, theta, base_params, n, n_models)

    # For a given identification method (first foler depth) and one sweep label (secnd depth folder)
    for id_label in identification_params:
        for sweep_label in SWEEP_TREES:
            compute_sweep_results(sweep_label, id_label)
    for id_label in identification_params:
        for sweep_label in SWEEP_LAYER:
            compute_sweep_results(sweep_label, id_label)

    # Plot figure
    plot_sweep(SWEEP_LAYER, "results_structure/sensitivity/sensitivity-layer.pdf")
    plot_sweep(SWEEP_TREES, "results_structure/sensitivity/sensitivity-trees.pdf")
