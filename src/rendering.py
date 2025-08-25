import numpy as np, matplotlib.pyplot as plt, json, os, pandas as pd
from tqdm import tqdm
from collections import defaultdict
from itertools import product

from matplotlib.lines import Line2D

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from general import *
from identification import get_heuristics


def plot_metric(metric, n_attackers, beam_sizes, 
                exh: Exhaustivness, model: Models,
                lucb_label=None,
                prefix="", ax=None, save=False):
    if ax is None:
        ax = plt.gca()
        show = True
    else: show = False
    if lucb_label is None: lab = ""
    else: lab = "-" + lucb_label
    att = {value: i for i, value in enumerate(n_attackers)}
    bs = {value: i for i, value in enumerate(beam_sizes)}
    for ls, struct in zip(("--", "-"), AlgoTypes):
        if os.path.isfile(prefix+f"results/{model.value}-{exh.value}/{struct.value}{lab}.json"):
            data = load_json(prefix+f"results/{model.value}-{exh.value}/{struct.value}{lab}.json")
        else:
            continue
        perfs = np.zeros((len(beam_sizes), len(n_attackers)))
        stds = np.zeros((len(beam_sizes), len(n_attackers)))
        for datum in data:
            if datum["beam_size"] == -1: continue
            perfs[bs[datum["beam_size"]], att[datum["n_attacker"]]] = datum[f"{metric}-avg"]
            stds[bs[datum["beam_size"]], att[datum["n_attacker"]]] = datum[f"{metric}-std"]
        for j, n_attacker in enumerate(n_attackers):
            ax.plot(np.array(beam_sizes) + j, perfs[:,j], 
                     color=f"C{j}", marker="x", ls=ls)
    
    ax.set_xlabel("Beam sizes")
    ax.set_ylabel(metric)

def plot_two_metrics(n_attackers, beam_sizes, metrics, title, prefix="../"):
    _, axes = plt.subplots(2, 5, figsize=(15,4.5), sharex=True,sharey="row")
    for measure, ax_line in zip(metrics,axes):
        line_gen = iter(ax_line)
        for model, ax in tqdm(list(zip((Models.BASE, Models.NON_BOOLEAN, Models.BLACK_BOX),line_gen))):
            plot_metric(measure, n_attackers, beam_sizes, 
                            Exhaustivness.FULL, model,
                            prefix="../", ax=ax)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.grid(axis="y")
        for lucb_label, ax in tqdm(list(zip(('lucb','naive'),line_gen))):
            plot_metric(measure, n_attackers, beam_sizes, 
                            Exhaustivness.FULL, Models.NOISY, lucb_label=lucb_label,
                            prefix="../", ax=ax)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.grid(axis="y")
    ax_gen = iter(axes[0])
    for ax, model in zip(axes[0,:3], (Models.BASE, Models.NON_BOOLEAN, Models.BLACK_BOX)):
        ax.set_title(model.value)
    for ax, lucb_label in zip(axes[0,3:5], ('lucb','naive')):
        ax.set_title(f"noisy - {lucb_label}")
    for ax, metric in zip(axes[:,0], metrics):
        ax.set_ylabel(metric)
    for ax in axes[1]: 
        if metrics[1] == "time":
            ax.set_yscale("log")
        ax.set_xticks(beam_sizes)
        ax.set_xlabel("Beam size")
        
    # Plot legend n_attacker
    for n_attacker in n_attackers:
        axes[1,0].plot([],[],label=f"n={n_attacker}")
    axes[1,0].legend(ncols=2)
    # Plot legend algo type
    for algo, ls in zip(AlgoTypes, ("--", "-")):
        axes[1,2].plot([],[],label=f"{algo.value}", ls=ls, c="grey")
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig(prefix+f"results/{title}.pdf")
    plt.show()
    


def plot_distributions(models, n_attackers, algo, exh, metric, prefix="../", base_size=2, plot_type="violin"):
    _, axes = plt.subplots(len(models),len(n_attackers), 
                           figsize=(base_size*len(n_attackers),base_size*len(models)), 
                           sharex=True, sharey='row', squeeze=False)
    
    for ax_line, (model, lucb_label) in tqdm(list(zip(axes, models))):
        for ax, n_attacker in zip(ax_line, n_attackers):
            plot_distribution(metric, n_attacker, algo, exh, 
                              model, lucb_label, ax, prefix, plot_type)
    for i, n_attacker in enumerate(n_attackers):
        axes[0,i].set_title(f"{n_attacker=}")
        axes[-1,i].set_xlabel("Beam sizes")
    for j, (model, lucb_label) in enumerate(models):
        lab = f" - {lucb_label}" if lucb_label else ""
        axes[j,0].set_ylabel(f"{model.value}{lab}\n\n{metric}")
    plt.tight_layout()
    plt.savefig(prefix+f"results/distributions-{algo.value}-{metric}.pdf")
    plt.show()

def plot_distribution(metric, n_attacker, algo, exh, model, lucb_label, ax=None, prefix="../", plot_type="violin"):
    if ax is None:
        ax = plt.gca()

    lab = f"-{lucb_label}" if lucb_label else ""
    data = load_json(prefix+f"results/{model.value}-{exh.value}/{algo.value}{lab}.json")
    values = {}
    for datum in data:
        if datum["beam_size"] == -1 or datum["n_attacker"] != n_attacker: continue
        values[datum["beam_size"]] = [res[metric] for res in datum["results"]]
    
    X_val = sorted(values.keys())
    Y = [values[x] for x in X_val]
    X = np.arange(len(X_val))
    if plot_type == "violin":
        ax.violinplot(Y, positions=X)
    elif plot_type == "box":
        ax.boxplot(Y, positions=X)
    ax.plot(X, np.mean(Y, axis=1), "x", c="black")
    ax.set_xticks(X, X_val)

def plot_noisy_params(Ns, prefix="../"):
    _, axes = plt.subplots(2,2, sharex=True)
    plot_noisy_params_metric(Ns, "F1", AlgoTypes.BASE, axes[0,0], prefix)
    plot_noisy_params_metric(Ns, "time", AlgoTypes.BASE, axes[1,0], prefix)
    plot_noisy_params_metric(Ns, "F1", AlgoTypes.STRUCTURED, axes[0,1], prefix)
    plot_noisy_params_metric(Ns, "time", AlgoTypes.STRUCTURED, axes[1,1], prefix)
    axes[0,0].set_title("Base algo")
    axes[0,1].set_title("Structured algo")
    for i in range(2):
        axes[1,i].set_xlabel("# sample per element")
    axes[0,1].legend()
    axes[0,0].set_ylabel("F1 score")
    axes[1,0].set_ylabel("time (s)")
    plt.tight_layout()
    plt.savefig(prefix+"results/params.pdf")
    plt.show()

def plot_noisy_params_metric(Ns, metric, algo, ax, prefix="../"):
    data = load_json(prefix+f"results/noisy-params/{algo.value}.json")
    perfs = defaultdict(lambda: np.zeros(len(Ns)))
    stds = defaultdict(lambda: np.zeros(len(Ns)))
    for datum in data:
        key = (datum['do_lucb'],datum['batch_size'])
        perfs[key][Ns.index(datum["N"])] = datum[f"{metric}-avg"]
        stds[key][Ns.index(datum["N"])] = datum[f"{metric}-std"]
    for key, value in perfs.items():
        label = f"batch size={key[1]}" if key[0] else "naive"
        ax.errorbar(Ns, value, yerr=stds[key], label=label, capsize=2, marker="x",elinewidth=1)
    ax.set_xticks(Ns)

def gather_data_smallest(prefix, algos, metrics, n_attackers, beam_sizes):
    all_perfs = {}
    all_stds = {}
    model = Models.BASE
    exh = Exhaustivness.SMALLEST
    for metric in tqdm(metrics):
        for algo_val in algos:
            data = load_json(prefix+f"results/{model.value}-{exh.value}/{algo_val}.json")
            perfs = np.zeros((len(beam_sizes), len(n_attackers)))
            stds = np.zeros((len(beam_sizes), len(n_attackers)))
            for datum in data:
                if datum["beam_size"] == -1: continue
                if datum["beam_size"] is None:
                    perfs[:, n_attackers.index(datum["n_attacker"])] = datum[f"{metric}-avg"]
                    stds[:, n_attackers.index(datum["n_attacker"])] = datum[f"{metric}-std"]
                else:
                    perfs[beam_sizes.index(datum["beam_size"]), n_attackers.index(datum["n_attacker"])] = datum[f"{metric}-avg"]
                    stds[beam_sizes.index(datum["beam_size"]), n_attackers.index(datum["n_attacker"])] = datum[f"{metric}-std"]
                all_perfs[f"{metric}-{algo_val}"] = perfs
                all_stds[f"{metric}-{algo_val}"] = stds
    return all_perfs, all_stds

def get_smallest_legend(ax):
    lines = []
    lines.append(Line2D([], [], label="Base algo", color="grey", ls="-", marker="x"))
    lines.append(Line2D([], [], label="Structured", color="grey", ls="--", marker="+"))
    lines.append(Line2D([], [], label="ILP", color="grey", ls="-.", marker="D"))
    return ax.legend(handles=lines, loc="lower left")

def plot_smallest_data(axes, metrics, Z, X, all_perfs, markers, lines, algos, xlabel, klabel):
    Z_ids = np.linspace(0,len(Z)-1,3,dtype=int)
    for ax, metric in zip(axes, metrics):
        for i, Z_i in enumerate(Z_ids):
            c_pad = 0 if xlabel == "beam sizes" else 3
            c = f"C{i+c_pad}"
            # Make the legends
            ax.plot([], label=f"{klabel}={Z[Z_i]}", color=c)
            
            for marker, ls, algo_val in zip(markers, lines, algos):
                # Do the plots
                perfs = all_perfs[f"{metric}-{algo_val}"]
                if perfs[Z_i].size != len(X): perfs = perfs.T
                if algo_val == "ILP": c = "grey"
                ax.plot(X, perfs[Z_i], color=c, marker=marker, ls=ls)
                       
            
        ax.set_xticks(X)
        
        if metric == 'time':
            ax.set_yscale("log")
    axes[0].legend(loc='upper left')
    axes[-1].set_xlabel(xlabel)

def plot_smallest(n_attackers, beam_sizes):
    exh = Exhaustivness.SMALLEST
    model = Models.BASE
    metrics = ("accuracy","time")
    prefix = "../"
    
    _, axes = plt.subplots(2,2,figsize=(7,4), sharey='row', sharex='col')
    legend_ax = axes[0,1]
    legend = get_smallest_legend(legend_ax)
    
    markers = ("x","+","D")
    lines = ("-", "--","-.")
    algos = (AlgoTypes.BASE.value,AlgoTypes.STRUCTURED.value,"ILP")

    all_perfs, all_stds = gather_data_smallest(prefix, algos, metrics, n_attackers, beam_sizes)

    plot_smallest_data(axes[:,0], metrics, n_attackers, beam_sizes, all_perfs, markers, lines, algos, "beam sizes","n")
    plot_smallest_data(axes[:,1], metrics, beam_sizes, n_attackers, all_perfs, markers, lines, algos, "# attackers", "b")

    legend_ax.add_artist(legend)
    for metric, ax_line in zip(metrics,axes):
        ax_line[0].set_ylabel(metric)
    plt.tight_layout()
    plt.savefig("../results/main-smallest.pdf")
    plt.show()

def plot_heuristics(prefix=""):
    measures = load_json(prefix+f"results/base-full/heuristics.json")
    X = list(get_heuristics(None).keys())
    X_pos = np.arange(len(X))
    Y = [measures[x] for x in X]
    Y_mean = [np.mean(measures[x]) for x in X]
    # Y_min = [np.std(measures[x]) for x in X]
    # Y_max = [np.std(measures[x]) for x in X]
    X = [x.replace("heuristic_", "") for x in X]
    plt.boxplot(Y, positions=X_pos)
    plt.plot(X_pos, Y_mean, "x")
    plt.ylabel("F1 scores")
    plt.xticks(X_pos, X, rotation=45)
    plt.savefig(prefix+"results/heuristics.pdf", bbox_inches='tight')
    plt.show()

def regression(x, y, degree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_pred)
    coefficients = model.named_steps['linearregression'].coef_
    return coefficients, r2, y_pred

def plot_regression(X, Y, Y_err, ax, i, z_label, degree):
    coefficients, r2, Y_pred = regression(X, Y, degree)
    if Y_err is None:
        ax.plot(X,Y, "x", c="black")
    else:
        ax.errorbar(X,Y, Y_err, c="black", ls="", marker="x")
    ax.plot(X,Y_pred, c="C0")
    ax.set_xticks(np.arange(X.size))
    ax.set_title(f"{z_label}\n$R^2={r2:.3f}$ - coef={coefficients[-1]:.2f}")

def plot_regressions(all_perfs, all_stds, algo_val, X, Z, z_lab, x_lab, transpose, do_ILP, 
                     n_col, n_row, degree, prefix="../", exh=Exhaustivness.SMALLEST):
    perf = all_perfs[f"time-{algo_val}"]
    stds = all_stds[f"time-{algo_val}"]
    acc = all_perfs[f"accuracy-{algo_val}"]
    if transpose:
        perf = perf.T
        stds = stds.T
        acc = acc.T
    _, axes = plt.subplots(n_row,n_col,figsize=(n_col*3,n_row*3), sharex=True)
    x = np.arange(len(X))

    ax_gen = iter(axes.flatten())
    for i, ax in zip(range(perf.shape[0]),ax_gen):
        y = perf[i]
        y_err = stds[i]
        plot_regression(x, y, y_err, ax, i, f"{z_lab}: {Z[i]}", degree)
        ax2 = ax.twinx()
        ax2.plot(x, acc[i], "--", c="grey")
        ax2.set_ylim(0,1.1)
        ax2.set_zorder(1)
        ax2.set_ylabel("Accuracy")
        ax.set_ylabel("Time (s)")
        
    for i in range(n_col):
        axes[1,i].set_xticks(x, X)
        axes[1,i].set_xlabel(x_lab)

    if do_ILP:
        plot_regression(x, all_perfs[f"time-ILP"][0], None, next(ax_gen), perf.shape[0], "ILP", degree)
    ax = next(ax_gen)
    ax.set_axis_off()
    ax.plot([], [], "x", color="black", label="Runtime")
    ax.plot([], [], "-", color="C0", label="Regression")
    ax.plot([], [], "--", color="grey", label="Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(prefix+f"results/smallest-time-vs-{x_lab}.pdf")
    plt.show()

def plot_regression_smallest(n_attackers, beam_sizes, metrics, prefix="../"):
    algos = (AlgoTypes.BASE.value,AlgoTypes.STRUCTURED.value,"ILP")
    
    all_perfs, all_stds = gather_data_smallest(prefix, algos, metrics, n_attackers, beam_sizes)

    plot_regressions(all_perfs, all_stds, "base_algo", 
                 n_attackers, beam_sizes, "beam size", "n attackers",
                 transpose=False, do_ILP=True, n_col=6, n_row=2, 
                 degree=2, prefix=prefix, exh=Exhaustivness.SMALLEST)

    plot_regressions(all_perfs, all_stds, "base_algo", 
                 beam_sizes, n_attackers, "n attackers", "beam size", 
                 transpose=True, do_ILP=False, n_col=5, n_row=2, 
                 degree=1, prefix=prefix, exh=Exhaustivness.SMALLEST)


# Plot increasing
def make_measures_increasing_one_setup(lines):
    measures = {
        "full-increase": 0,
        "increase-0.01": 0,
        "increase-0.05": 0,
        "increase-0.1": 0,
        "increase-overall": 0
    }
    for line in lines:
        measures["full-increase"] += check_increase(line, 0)
        measures["increase-0.01"] += check_increase(line, 0.01)
        measures["increase-0.05"] += check_increase(line, 0.05)
        measures["increase-0.1"] += check_increase(line, 0.1)
        measures["increase-overall"] += check_increase_overall(line)
    for m in measures.keys():
        measures[m] = float(measures[m] / len(lines))
    return measures

def check_increase(line, threshold):
    return np.all(line[1:]-line[:-1] >= -threshold)

def check_increase_overall(line):
    return line[-1] - line[0] >= 0

def get_lines_increasing(data, beam_sizes, n_attacker, metric):
    lines = defaultdict(lambda: np.zeros(len(beam_sizes)))
    bs2id = {bs: i for i, bs in enumerate(beam_sizes)}
    for d in data:
        if d["n_attacker"] == n_attacker:
            for value in d["results"]:
                context = tuple(value["context"])
                bsId = bs2id[d["beam_size"]]
                lines[context][bsId] = value[metric]
    return np.array(list(lines.values()))

def make_measures_increasing(n_attacker, algo, exh, model, prefix, show=False):
    with open(prefix+f"results/{model.value}-{exh.value}/{algo.value}.json") as file:
        data = json.load(file)
    metric = "F1" if exh == Exhaustivness.FULL else "accuracy"
    bs = beam_sizes if exh == Exhaustivness.FULL else beam_sizes_smallest
    lines = get_lines_increasing(data, bs, n_attacker, metric)
    if show:
        plt.plot(bs, lines.T)
        plt.xlabel("Beam Size")
        plt.ylabel(metric)
        plt.xticks(bs)
        plt.title(f"{model.value}-{algo.value}-{exh.value}-n={n_attacker}")
        plt.show()
    return make_measures_increasing_one_setup(lines)

def render_table_increasing(prefix):
    all_measures = {}
    for algo, exh, model in tqdm(list(product(AlgoTypes, Exhaustivness, Models))):
        ns = n_attackers if exh == Exhaustivness.FULL else n_attackers_smallest
        for n in ns:
            key = f"{model.value}-{algo.value}-{exh.value}-{n=}"
            try: all_measures[key] = make_measures_increasing(
                n, algo, exh, model, "../"
                )
            except: pass
    return pd.DataFrame(all_measures).T

def plot_metric_short(metric, n_attackers, beam_sizes, 
                exh: Exhaustivness, prefix="", ax=None, save=False, prop_cycle=0):
    model = Models.BASE
    if ax is None:
        ax = plt.gca()
        show = True
    else: show = False
    att = {value: i for i, value in enumerate(n_attackers)}
    bs = {value: i for i, value in enumerate(beam_sizes)}
    options = [(AlgoTypes.BASE.value, "-", "x"), 
               (AlgoTypes.STRUCTURED.value,"--", "+")]
    if exh == Exhaustivness.SMALLEST:
        options.append(("ILP", "-.", "D"))
    for algo, ls, marker in options:
        data = load_json(prefix+f"results/{model.value}-{exh.value}/{algo}.json")
        perfs = np.zeros((len(beam_sizes), len(n_attackers)))
        for datum in data:
            if datum["n_attacker"] not in att: continue
            if algo == "ILP":
                perfs[:, att[datum["n_attacker"]]] = datum[f"{metric}-avg"]
            elif datum["beam_size"] in bs:
                perfs[bs[datum["beam_size"]], att[datum["n_attacker"]]] = datum[f"{metric}-avg"]
        for j, n_attacker in enumerate(n_attackers):
            ax.plot(np.array(beam_sizes) + j, perfs[:,j], 
                     color=f"C{prop_cycle+j}", marker=marker, ls=ls)
    
    ax.set_xlabel("Beam sizes")
    ax.set_ylabel(metric)

def plot_metrics_short(prefix="../"):
    sub_na_small = (4,11,20)
    sub_bs_small = (1, 22, 44, 66, 88, 100)
    _, axes = plt.subplots(2, 3, figsize=(9,4.5), sharey="row", sharex="col")
    plot_metric_short("F1", n_attackers, beam_sizes, Exhaustivness.FULL, 
                      prefix=prefix, ax=axes[0,0])
    plot_metric_short("accuracy", sub_na_small, sub_bs_small, Exhaustivness.SMALLEST, 
                      prefix=prefix, ax=axes[0,1], prop_cycle=3)
    plot_metric_short("time", n_attackers, beam_sizes, Exhaustivness.FULL, 
                      prefix=prefix, ax=axes[1,0])
    plot_metric_short("time", sub_na_small, sub_bs_small, Exhaustivness.SMALLEST, 
                      prefix=prefix, ax=axes[1,1], prop_cycle=3)
    
    
    for ax in axes.flatten():
        ax.grid(axis="y")
    
    axes[1,1].set_ylabel("")
    axes[0,0].set_xlabel("")
    axes[0,1].set_xlabel("")
    
    axes[1,0].set_yscale("log")
    axes[1,1].set_yscale("log")
    
    # Legend
    for j, n_attacker in enumerate(n_attackers + sub_na_small):
        axes[0,2].plot([],[],c=f"C{j}",label=f"n={n_attacker}")
    axes[0,2].plot([],[],c="grey", marker="x", label="Base algo")
    axes[0,2].plot([],[],c="grey",ls="--", marker="+", label="ISI algo")
    axes[0,2].plot([],[],c="grey",ls="-.", marker="D", label="ILP")
    axes[0,2].legend(loc="center left")
    axes[0,2].set_axis_off()
    axes[1,2].set_axis_off()
    axes[1,0].set_xticks(beam_sizes)
    axes[1,1].set_xticks(sub_bs_small)
    
    axes[0,0].set_title("Full identification")
    axes[0,1].set_title("Smallest identification")
    plt.subplots_adjust(hspace=0.1)
    # plt.tight_layout()
    plt.savefig(prefix+"results/fig_short.pdf")

def plot_distribution_short(metric, n_attacker, algo, exh, model, lucb_label, beam_sizes, ax=None, prefix="../", plot_type="violin"):
    if ax is None:
        ax = plt.gca()

    lab = f"-{lucb_label}" if lucb_label else ""
    data = load_json(prefix+f"results/{model.value}-{exh.value}/{algo.value}{lab}.json")
    values = {}
    for datum in data:
        if datum["beam_size"] == -1 or datum["n_attacker"] != n_attacker: continue
        values[datum["beam_size"]] = [res[metric] for res in datum["results"]]
    
    X_val = sorted(beam_sizes)
    Y = [values[x] for x in X_val]
    X = np.arange(len(X_val))
    if plot_type == "violin":
        ax.violinplot(Y, positions=X)
    elif plot_type == "box":
        ax.boxplot(Y, positions=X)
    ax.plot(X, np.mean(Y, axis=1), "x", c="black")
    ax.set_xticks(X, X_val)

def plot_distributions_short(n_attackers, exh, beam_sizes, prefix="../", plot_type="violin"):
    _, axes = plt.subplots(4,3, sharex=True, sharey='row', squeeze=False, figsize=(6,8))
    model = Models.BASE
    if exh == Exhaustivness.FULL: m = "F1"
    else: m = "accuracy"
    lucb_label = ""
    for i, n_attacker in enumerate(n_attackers):
        plot_distribution_short(m, n_attacker, AlgoTypes.BASE, exh, 
                          model, lucb_label, beam_sizes, axes[0,i], prefix, plot_type)
        plot_distribution_short("time", n_attacker, AlgoTypes.BASE, exh, 
                          model, lucb_label, beam_sizes, axes[1,i], prefix, plot_type)
        plot_distribution_short(m, n_attacker, AlgoTypes.STRUCTURED, exh, 
                          model, lucb_label, beam_sizes, axes[2,i], prefix, plot_type)
        plot_distribution_short("time", n_attacker, AlgoTypes.STRUCTURED, exh, 
                          model, lucb_label, beam_sizes, axes[3,i], prefix, plot_type)
    
    axes[0,0].set_ylabel(f"Base algo\n\n{m}")
    axes[1,0].set_ylabel("Base algo\n\ntime (s)")
    
    axes[2,0].set_ylabel(f"ISI algo\n\n{m}")
    axes[3,0].set_ylabel("ISI algo\n\ntime (s)")
    
    for i in range(3): 
        axes[-1,i].set_xlabel("Beam size")
        axes[0,i].set_title(f"n={n_attackers[i]}")
    
    
    plt.tight_layout()
    plt.savefig(prefix+f"results/distributions_{exh.value}_short.pdf")
    plt.show()