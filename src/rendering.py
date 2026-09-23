import numpy as np, timeit, time, matplotlib.pyplot as plt, json, os, pandas as pd
import matplotlib.ticker as ticker
from tqdm import tqdm
from itertools import product
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from general import *
from sanity import *
from experiments import exps, exps_reg, exps_smallest, AlgoTypes, Models

w,h=3.1,2.1
label_map = {"structured":"ISI", "bs":"b", "n":"n", "base_algo":"MBS", "ILP":"ILP"}

# == Retrieve values ==
def get_values(exps, metrics, folder="results/"):
    table = []
    columns = ["exh", "algo", "model", "lucb_label", "heuristic", "n", "bs", "u", "seed", "time","n_calls"] + metrics
    for exh, model, algo, _, _, heuristics, lucb_label, _ in exps:
        file_name = get_file_name(exh, model, algo, heuristics, lucb_label)
        try:
            data = load_json(folder+file_name)
            print(f"-> {file_name}")
        except FileNotFoundError: 
            print(f"-x {file_name}")
            continue
        for datum in data:
            bs, n = datum["beam_size"], datum["n_attacker"]
            heuristic = datum.get("heuristic")
            for res in datum["results"]:
                u = tuple(res["context"])
                if "seed" in res: seed = res["seed"]
                else: seed = None
                n_calls = res.get("n_calls")
                row = [
                    exh.value, algo.value, model.value, lucb_label, heuristic, n, bs, u, seed, res["time"], n_calls
                ]
                for metric in metrics:
                    if "metrics" not in res or metric not in res["metrics"]:
                        row.append(None)
                    else:
                        row.append(res["metrics"][metric])
                table.append(row)
    return pd.DataFrame(table, columns=columns)

def get_values_ILP(metrics, folder):
    table = []
    columns = ["exh", "algo", "model", "lucb_label", "heuristic", "n", "bs", "u", "seed", "time","n_calls"] + metrics
    file_name = "base-smallest/ILP.json"
    try:
        data = load_json(folder+file_name)
        print(f"-> {file_name}")
    except FileNotFoundError: 
        print(f"-x {file_name}")
        return pd.DataFrame(table, columns=columns)
        
    heuristic, lucb_label, seed, n_calls = (None,) * 4
    for datum in data:
        bs, n = datum["beam_size"], datum["n_attacker"]
        seed = None
        for res in datum["results"]:
            u = tuple(res["context"])
            n_calls = res.get("n_calls")
            row = [
                Exhaustivness.SMALLEST.value, "ILP", Models.BASE.value, 
                lucb_label, heuristic, n, bs, u, seed, res["time"], n_calls
            ]
            for metric in metrics:
                if "metrics" not in res or metric not in res["metrics"]:
                    row.append(None)
                else:
                    row.append(res["metrics"][metric])
            table.append(row)
    return pd.DataFrame(table, columns=columns)

# == Main figure ==
def plot_general(df, models, metrics, no_share=None, agg="median", folder="figures/", beam_sizes=None):
    rows, cols = len(metrics),len(models)
    _, axes = plt.subplots(rows, cols, figsize=(w*cols,h*rows), sharex=True)
    
    for row in range(rows):
        if no_share is None or row in no_share:
            continue
        ref = axes[row, 0]
        for c in range(1, cols):
            axes[row, c].sharey(ref)
            axes[row, c].tick_params(labelleft=False)
            
    df = df[(df.exh=="full") & df.heuristic.isna()]
    for j, (model, lucb_label) in enumerate(models):
        for i, metric in enumerate(metrics):
            for algo in set(df.algo):
                for c, n in enumerate(sorted(set(df.n))):
                    index = (df.algo == algo) & (df.model==model) & (df.n == n) & df.heuristic.isna()
                    if lucb_label:
                        index &= (df.lucb_label == lucb_label)
                    if agg == "median":
                        df_ = df[index].groupby(["bs"], as_index=True)[metric].median()
                        up = df[index].groupby(["bs"], as_index=True)[metric].quantile(.75)
                        low = df[index].groupby(["bs"], as_index=True)[metric].quantile(.25)
                        dev = [df_.array-low.array, up.array-df_.array]
                    else:
                        df_ = df[index].groupby(["bs"], as_index=True)[metric].mean()
                        dev = df[index].groupby(["bs"], as_index=True)[metric].std()
                    
                    x = np.arange(df_.index.size)
                    if algo == AlgoTypes.STRUCTURED.value: ls = "--"
                    else: ls = "-"
                    axes[i,j].errorbar(x, df_.array, yerr=dev, marker="x",ls=ls, c=f"C{c}")
        axes[0,j].grid(axis="y")
        axes[1,j].grid(axis="y")
        axes[1,j].set_xlabel("Beam size")
        axes[0,j].set_title(model + "-"*bool(lucb_label)+lucb_label)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # always use 10^n form
        axes[1,j].yaxis.set_major_formatter(formatter)

    for c, n in enumerate(sorted(set(df.n))):
        axes[1,0].plot([], c=f"C{c}", label=f"{n=}")
    for algo in set(df.algo):
        if algo == AlgoTypes.STRUCTURED.value: ls = "--"
        else: ls = "-"
        axes[1,0].plot([], c="grey", ls=ls, label=label_map[algo])
    axes[1,0].legend()

    if beam_sizes is not None:
        for ax in axes[:,-1]:
            ax.set_xticks(x, beam_sizes)
    
    for ax, metric in zip(axes.T[0], metrics): ax.set_ylabel(metric)
    plt.tight_layout()
    plt.savefig(folder+"-".join(metrics)+f"-{agg}.pdf")
    plt.show()

# == Spaghetti plot ==
def plot_spaguetti(df, models, algo, metric):
    df = df[(df.exh=="full") & df.heuristic.isna()]
    algos = sorted(set(df.algo))
    ns = sorted(set(df.n))
    _, axes = plt.subplots(len(ns), len(models), figsize=(w*len(models),h*len(ns)), 
                           sharex=True, sharey="row")
    for j, (model, lucb_label) in tqdm(list(enumerate(models))):
        for i, n in enumerate(ns):
            index = (df.algo == algo.value) & (df.model==model) & (df.n == n) & df.heuristic.isna()
            if lucb_label:
                index &= (df.lucb_label == lucb_label)
            x = np.arange(len(set(df[index].bs)))
            for c, u in enumerate(sorted(set(df[df.n==n].u))):
                df_ = df[index & (df.u == u)].groupby(["bs"], as_index=True)[metric].mean() + .001*c
                # df_.plot(style="x-", ax=axes[i,j], c=f"C{c}", alpha=.5)
                axes[i,j].plot(x, df_.array, marker="x", ls="-", c=f"C{c}", alpha=.5)
            df_ = df[index].groupby(["bs"], as_index=True)[metric].mean()
            axes[i,j].plot(x, df_.array, marker="x", ls="-", c=f"black")
            # df_.plot(style="x-", ax=axes[i,j], c=f"black")
            axes[i,j].grid(axis="y")
            axes[i,j].set_xticks(x, df_.index)
        axes[0,j].set_title(model + "-"*bool(lucb_label)+lucb_label)
    plt.tight_layout()
    plt.show()

# == Tradeff figure ==
def plot_tradeoff(df, exh, algo, n, quality_metric, eff_metric="n_calls", ax=None):
    if ax is None: 
        plt.figure(figsize=(w*1.2,h))
        ax = plt.gca()
    
    index = (
        (df.exh==exh) & (df.heuristic.isna()) & (df.algo == algo) & (df.model == "base") & (df.n == n)
    )
    df[[quality_metric, eff_metric]] = df[[quality_metric, eff_metric]].apply(pd.to_numeric, errors='coerce')
    group = df[index].groupby(["bs"], dropna=True)
    stds = group[[quality_metric, eff_metric]].std()
    lows = group[[quality_metric, eff_metric]].quantile(.25)
    highs = group[[quality_metric, eff_metric]].quantile(.75)
    df_mean = group[[quality_metric, eff_metric]].mean()
    df_median = group[[quality_metric, eff_metric]].median()
    
    t_std = stds[eff_metric] #/ 2
    q_std = stds[quality_metric] #/ 2
    t_high = highs[eff_metric]
    t_low = lows[eff_metric]
    q_high = highs[quality_metric]
    q_low = lows[quality_metric]
    if quality_metric == "accuracy":
        df_ = df_mean
    else:
        df_ = df_median
    bs = df_.index.array
    t = df_[eff_metric].array
    q = df_[quality_metric].array
    if quality_metric == "accuracy":
        ax.errorbar(t, q, xerr=t_std, ls='--',marker="x", capsize=2, ecolor="grey")
    else:
        ax.errorbar(t, q, xerr=[t-t_low, t_high-t], yerr=[q-q_low, q_high-q], ls='--',marker="x", capsize=2, ecolor="grey")
    for xi, yi, label in zip(t, q, bs):
        if label in (2, 4, 8):
            ax.text(xi, yi, label, fontsize=8, ha='right', va='bottom')
        else:
            if label == 256 and exh == "smallest": continue
            ax.text(xi, yi, label, fontsize=8, ha='left', va='top', rotation=60)
    # ax.set_ylim(min(q)-12, max(q) + .1*(max(q)-min(q)))
    ax.set_xlim(min(t)/2., max(t)*2.)
    ax.set_xlabel(eff_metric)
    ax.set_ylabel(quality_metric)
    ax.set_xscale("log")

def plot_full_tradeoffs(df, folder="figures/"):
    comps = (
        ("full", "base_algo", 2), 
        ("full", "structured", 10),
        ("smallest", "base_algo", 7)
    )
    for i, (exh, algo, n) in enumerate(comps):
        metric = "dice" if exh == "full" else "accuracy"
        plot_tradeoff(df, exh, algo, n, metric)
        lab = "ISI" if algo == "structured" else algo
        plt.savefig(folder + f"tradeoffs-{exh}-{lab}-n={n}.pdf", bbox_inches="tight")
        plt.show()

# == Heuristic figure ==
def plot_heuristic(df, folder="figures/"):
    _, axes = plt.subplots(1,2, figsize=(2*w, 1.3*h))
    for algo, ax in zip(("base_algo", "structured"), axes):
        df_ = df[~df.heuristic.isna() & (df.algo == algo)]
        xticks = df_.heuristic.unique()
        for i, psi in enumerate(xticks):
            values = df_[df_.heuristic == psi].dice.array
            x = np.linspace(i-.1,i+.1, len(values))
            x = np.full(len(values), i)
            ax.scatter(x, values, alpha=.2, color="grey")
            ax.scatter([i], [values.mean()], color="black", marker="+")
            ax.text(i, values.mean(), f"{values.mean():.0f}%", va="top")
        ax.set_xticks(np.arange(len(xticks)), xticks, rotation=45)
    axes[0].set_title(f"Beam Search - n=2 - b=64")
    axes[1].set_title(f"ISI - n=10 - b=8")
    axes[0].set_ylabel("Dice")
    plt.tight_layout()
    plt.savefig(folder+"heuristic_plot.pdf")
    plt.show()

# == Smallest identification figure ==
def show_smallest_comparison(df, beam_sizes, ax=None):
    if ax is None: ax = plt.gca()
    comps = [("structured", bs) for bs in beam_sizes]
    comps += [("base_algo", bs) for bs in beam_sizes]
    comps += [("ILP", None)]
    
    for c, (algo, bs) in enumerate(comps):
        index = (df.exh == "smallest") & (df.algo == algo)
        label = label_map[algo]
        if bs is not None: 
            index &= (df.bs == bs)
            label += f"{bs}"
        ax.plot([],label=label, c=f"C{c}", marker="x")
        df_ = df[index].groupby(["n"], as_index=False)[["time", "accuracy"]].mean()
        ns = df_.index.array
        for i in range(len(ns)-1):
            ni = ns[i]
            nj = ns[i+1]
            xi = len(get_SMK_V(ni))
            xj = len(get_SMK_V(nj))
            yi = df_.loc[i, "time"]
            yj = df_.loc[i+1, "time"]
            ai = abs(df_.loc[i, "accuracy"] - 100) < .1
            aj = abs(df_.loc[i+1, "accuracy"] - 100) < .1
            ls = "-" if (ai and aj) else '--'
            ax.plot([xi, xj], [yi, yj], c=f"C{c}", ls=ls, marker="x")
    ax.set_yscale("log")
    ax.set_ylabel("time (s)")
    ax.set_xlabel("|V|")
    ax.legend(loc="lower right", ncols=2)

def show_smallest_perf(df, beam_sizes, ax=None):
    if ax is None: ax = plt.gca()
    comps = [("structured", bs) for bs in beam_sizes]
    comps += [("base_algo", bs) for bs in beam_sizes]
        
    for c, (algo, bs) in enumerate(comps):
        index = (df.exh == "smallest") & (df.algo == algo) & (df.bs == bs)
        df_ = df[index].groupby(["n"])["accuracy"].mean()
        x = np.array([len(get_SMK_V(n)) for n in df_.index])
        ax.plot(x+.5*c, df_.array+.5*c)#, label=f"b={bs}")
        label = label_map[algo]
        if bs is not None: 
            index &= (df.bs == bs)
            label += f"{bs}"
        ax.plot([],label=label, c=f"C{c}", marker="x")
    ax.legend()
    ax.set_xlabel("|V|")
    ax.set_ylabel("Accuracy")

def plot_smallest(df, folder="figures/"):
    _, axes = plt.subplots(1,2, figsize=(3*w, 1.5*h))
    show_smallest_comparison(df, (4,32,256), ax=axes[0])
    show_smallest_perf(df, (4,32,256), ax=axes[1])
    axes[0].set_title("Time against system size")
    axes[1].set_title("Accuracy against system size (MBS)")
    plt.tight_layout()
    plt.savefig(folder+"smallest_fig.pdf")
    plt.show()

# == Regression figures ==
def regression(x, y, degree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_pred)
    coefficients = model.named_steps['linearregression'].coef_
    return coefficients, r2, y_pred

candidates = {
    "log n": lambda n: np.log(n),
    "n": lambda n: n,
    "n²": lambda n: n**2,
    "n³": lambda n: n**3,
    "√n": lambda n: np.sqrt(n),
}

def fit_model(x, y, g):
    X = g(x).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    return r2, y_pred, (model.coef_[0], model.intercept_)

def find_model(x,y, verbose=False):
    results = {}
    
    for name, fn in candidates.items():
        results[name] = fit_model(x, y, fn)
    sorted_results = sorted(results.items(), key=lambda kv: -kv[1][0])
    
    if verbose:
        print("\nEmpirical Complexity Fit:")
        print("--------------------------")
        for name, (r2, a, b) in sorted_results:
            print(f"{name:10s}  R² = {r2:.6f}   a = {a:.6e}   b = {b:.6e}")
    return sorted_results[0]

def plot_all_regressions(df, folder="figures/"):
    for exh in ("smallest", "full"):
        _, axes = plt.subplots(2,2, figsize=(2*w*1.5,2*h*1.5))
    
        axes[0,0].set_title("MBS")
        axes[0,1].set_title("ISI")
    
        ns = (2,4,6,8,10,12,14)
        plot_reg_x_per_z(df, ns, "n", "bs", exh, "base_algo", axes[0,0])
        plot_reg_x_per_z(df, ns, "n", "bs", exh, "structured", axes[0,1])
    
        bss = (2,4,8,16,32,64,128,256)
        plot_reg_x_per_z(df, bss, "bs", "n", exh, "base_algo", axes[1,0])
        plot_reg_x_per_z(df, bss, "bs", "n", exh, "structured", axes[1,1])
    
        for ax in axes.flatten():
            ax.legend(ncols=2)
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))  
            ax.yaxis.set_major_formatter(formatter)
        for ax in axes[0,:]: ax.set_xlabel("beam size")
        for ax in axes[1,:]: ax.set_xlabel("|V|")
        for ax in axes[:,0]: ax.set_ylabel("n_calls")
        plt.tight_layout()
        plt.savefig(folder+f"regressions-{exh}.pdf")
        plt.show()

def plot_reg_x_per_z(df, zs, z_label, x_label, exh, algo, ax=None, y_label="n_calls"):
    if ax is None: ax = plt.gca()
    for i, z in enumerate(zs):
        index = (
            (df.exh==exh) & (df.algo==algo) & (df[z_label]==z) & df.heuristic.isna() & (df.bs != -1)
        )
        
        df_ = df[index].groupby([x_label])[[y_label]].mean()
        if not df_.index.size: continue
        X, Y = df_.index.array, df_[y_label].array
        fit, (r2, Y_pred, coefs) = find_model(X,Y)
        ax.plot(X+.1*i,Y, "x", c=f"C{i}", ls='-', label=f"{z_label}={int(z)}: {fit} {r2:.0%}")
        ax.plot(X+.1*i,Y_pred, c=f"C{i}",ls='--')

# == Tables comparing the algorithm improvements ==
def fmt(m, s):
    return rf"{m:.0f} \textcolor{{gray}}{{\scriptsize$\pm$ {s:.0f}}}"
    
def save_latex(mean, std, name, folder):
    latex_df = mean.copy()
    for c in mean.columns:
        latex_df[c] = [
            fmt(m, s)
            for m, s in zip(mean[c], std[c])
        ]
    latex_df.columns.name = None
    latex = latex_df.to_latex(escape=False, column_format="cc|cc")
    latex = latex.replace(r"\multirow[t]{", r"\multirow{")
    latex_document = r"""
    \documentclass{article}
    \usepackage{booktabs}
    \usepackage{multirow}
    \usepackage{xcolor}
    \begin{document}
    
    % The table:
    """ + latex + r"""
    
    \end{document}
    """
    
    with open(folder + name + ".tex", "w") as f:
        f.write(latex)

def format_df(mean, std):
    formatted = mean.copy()
    for c in mean.columns:
        formatted[c] = mean[c].combine(
            std[c],
            lambda m, s: f"{m:.0f} ± {s:.0f}"
        )
    return formatted

def compare_algo(df, algo_comp, metric, name=None, folder="tables/"):
    if algo_comp == "ISI": 
        ref = "model"
        labels_ref = ["base", "non-boolean"]
        comp = "algo"
        labels_comp = ["base_algo", "structured"]
        group = "model"
        index = ((df.model=="base")|(df.model=="non-boolean"))&(df.heuristic.isna())&(df.exh=="full")
    else: 
        ref = "algo"
        labels_ref = ["base_algo", "structured"]
        comp = "lucb_label"
        labels_comp = ["naive", "lucb"]
        group = "lucb_label"
        index = (df.model=="noisy")&(df.heuristic.isna())&(df.exh=="full")
    if metric == "n_calls":
        do_comp = lambda df_: (df_[labels_comp[0]] - df_[labels_comp[1]]) / df_[labels_comp[0]] * 100
    else:
        do_comp = lambda df_: df_[labels_comp[1]] - df_[labels_comp[0]]
    df_ = df[index].groupby(["algo", "n", "bs", group, "u"], as_index=False)[metric].mean()
    df_ = df_.set_index(["n", "bs", ref, "u"]).pivot(columns=comp, values=metric)
    df_ = do_comp(df_)
    df_ = df_.to_frame().reset_index().pivot(columns=ref, index=["n", "bs", "u"], values=0)
    mean = df_.groupby(["n", "bs"])[labels_ref].mean()
    std = df_.groupby(["n", "bs"])[labels_ref].std()
    if name is not None: save_latex(mean, std, name, folder)
    return format_df(mean, std)

# == Retrieve and show some relevant number to put in article ==
def locate_text_numbers(df):
    index = ((df.model=="base")&(df.n==2)&(df.heuristic.isna())&(df.algo=="base_algo")&(df.exh=="full"))
    df_ = df[index].groupby(["bs"]).dice.mean()
    print(f"base algo, base model, full, n=2:         bs:8->16:  +{df_.loc[16] - df_.loc[8]:.1f} dice points")
    
    index = ((df.model=="base")&(df.n==10)&(df.heuristic.isna())&(df.algo=="structured")&(df.exh=="full"))
    df_ = df[index].groupby(["bs"]).dice.mean()
    print(f"ISI algo, base model, full, n=10:         bs:8->16:  +{df_.loc[16] - df_.loc[8]:.1f} dice points")
    
    index = ((df.model=="base")&(df.n==5)&(df.heuristic.isna())&(df.bs==4)&(df.exh=="full"))
    df_ = df[index].groupby(["algo"]).dice.mean()
    print(f"base model, full, n=10, bs=4:             base->ISI: +{df_.loc['structured'] - df_.loc['base_algo']:.1f} dice points")

    for n in (2, 5, 10):
        index = ((df.model=="base")&(df.n==n)&(df.bs==-1)&(df.heuristic.isna())&(df.exh=="exact")&(df.algo=="structured"))
        calls = df[index].n_calls
        times = df[index].time
        print(f"base model, full, n={n},  bs=-1: {calls.mean():.2f}±{calls.std():.2f} calls / {times.mean():.2f}±{times.std():.2f} calls")


if __name__ == "__main__":
    metrics = ["F1", "accuracy", "jaccard", "dice"]
    df = get_values(exps, metrics, "results/")

    df_reg = get_values(exps_reg, ["accuracy"], folder="results_reg/")

    df_ilp = get_values_ILP(["accuracy"], "results_smallest/")
    df_smallest = get_values(exps_smallest, ["accuracy"], folder="results_smallest/")
    df_smallest = pd.concat([df_smallest, df_ilp], ignore_index=True)

    df.F1 *= 100
    df.dice *= 100
    df.accuracy *= 100
    df_reg.accuracy *= 100
    df_smallest.accuracy *= 100

    models = (
    (Models.BASE.value, ""), 
    (Models.NON_BOOLEAN.value, ""), 
    (Models.NOISY.value, "lucb"), 
    (Models.NOISY.value, "naive"))

    # plot_general(df, models, ["dice", "n_calls"], [1], "median", "figures/")
    plot_full_tradeoffs(pd.concat([df,df_reg]), "figures/")
    # plot_smallest(df_smallest)
    # plot_all_regressions(df_reg)
    # plot_heuristic(df)


    # """Sanity"""
    # print("Sanity checks")
    # lines, index = run_sanity_checks(sanity_checks)
    # plot_sanity_checks(lines, index)

    """Comparison"""
    # print("Compare ISI/MBS on n_calls")
    # print(compare_algo(df, "ISI", "n_calls", "call_gain_ISI", "tables/"))
    # print("Compare ISI/MBS on dice")
    # print(compare_algo(df, "ISI", "dice", "dice_gain_ISI", "tables/"))
    # print("Compare Naive/LUCB on n_calls")
    # print(compare_algo(df, "lucb", "n_calls", "call_gain_LUCB", "tables/"))
    # print("Compare Naive/LUCB on dice")
    # print(compare_algo(df, "lucb", "dice", "dice_gain_LUCB", "tables/"))
    # locate_text_numbers(df)

    # """Exact identification"""
    # print("Mean time (s) and number of calls for the exact identification")
    # print(df[df.exh == "exact"].groupby("n")[["time","n_calls"]].mean())

    # print("Std time (s) and number of calls for the exact identification")
    # print(df[df.exh == "exact"].groupby("n")[["time","n_calls"]].std())