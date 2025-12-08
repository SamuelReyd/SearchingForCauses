import numpy as np, timeit, time, matplotlib.pyplot as plt, json, os, pandas as pd
import matplotlib.ticker as ticker
from tqdm import tqdm
from itertools import product
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from general import *

w,h=3.1,2.1

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
def plot_metric_mean(df, models, metrics, no_share=None, folder="figures/"):
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
                    df_ = df[index].groupby(["bs"], as_index=True)[metric].mean()
                    std = df[index].groupby(["bs"], as_index=True)[metric].std()/2
                    x = np.arange(df_.index.size)
                    if algo == AlgoTypes.STRUCTURED.value: ls = "--"
                    else: ls = "-"
                    axes[i,j].errorbar(x, df_.array, yerr=std, marker="x",ls=ls, c=f"C{c}")
                    axes[i,j].set_xticks(x,df_.index)
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
        axes[1,0].plot([], c="grey", ls=ls, label=algo)
    axes[1,0].legend()
    
    for ax, metric in zip(axes.T[0], metrics): ax.set_ylabel(metric)
    plt.tight_layout()
    plt.savefig(folder+"-".join(metrics)+"-mean.pdf")
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
def plot_tradeoff(df, exh, algo, n, quality_metric, ax=None):
    if ax is None: ax = plt.gca()
    index = (
        (df.exh==exh) & (df.heuristic.isna()) & (df.algo == algo) & (df.model == "base") & (df.n == n)
    )
    # print(df[index])
    group = df[index].groupby(["bs"], dropna=False)
    stds = group[[quality_metric,"time"]].std()
    df_ = group.mean([quality_metric, "time"])
    
    t_std = stds.time / 2
    q_std = stds[quality_metric] / 2
    # print(df_)
    bs = df_.index.array
    t = df_.time.array
    q = df_[quality_metric].array
    ax.errorbar(t, q, xerr=t_std, yerr=q_std, ls='--',capsize=2, ecolor="grey")
    for xi, yi, label in zip(t, q, bs):
        ax.text(xi, yi, label, fontsize=9, ha='right', va='bottom')
    ax.set_ylim(min(q)-12, max(q) + .1*(max(q)-min(q)))
    ax.set_xlabel("time (s)")
    ax.set_ylabel(quality_metric)
    ax.set_xscale("log")

def plot_full_tradeoffs(df, folder="figures/"):
    # _, axes = plt.subplots(1,3, figsize=(3*w, 1.5*h))
    comps = (
        ("full", "base_algo", 2), 
        ("full", "structured", 10),
        ("smallest", "base_algo", 7)
    )
    for i, (exh, algo, n) in enumerate(comps):
        metric = "dice" if exh == "full" else "accuracy"
        plot_tradeoff(df, exh, algo, n, metric)
        lab = "ISI" if algo == "structured" else algo
        # axes[i].set_title(f"{exh} - {lab} - n={n}")
        plt.savefig(folder + f"tradeoffs-{exh}-{lab}-n={n}.pdf")
        plt.show()
    # plt.tight_layout()
    # plt.savefig(folder + "tradeoffs.pdf")
    # plt.show()

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
        label = "ISI" if algo == "structured" else algo
        if bs is not None: 
            index &= (df.bs == bs)
            label += f" - b={bs}"
        ax.plot([],label=label, c=f"C{c}", marker="x")
        df_ = df[index].groupby(["n"], as_index=False).mean(["time", "accuracy"])
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
    ax.legend(loc="center right")

def show_smallest_perf(df, beam_sizes, ax=None):
    if ax is None: ax = plt.gca()
    for i,bs in enumerate((2,4,8,16,32,64)):
        index = (df.exh == "smallest") & (df.algo == "base_algo") & (df.bs == bs)
        df_ = df[index].groupby(["n"])["accuracy"].mean()
        x = np.array([len(get_SMK_V(n)) for n in df_.index])
        ax.plot(x+.8*i, df_.array, label=f"b={bs}")
    ax.legend()
    ax.set_xlabel("|V|")
    ax.set_ylabel("Accuracy")

def plot_smallest(df, folder="figures/"):
    _, axes = plt.subplots(1,2, figsize=(3.5*w, 2*h))
    show_smallest_comparison(df, (4,32,256), ax=axes[0])
    show_smallest_perf(df, (2,4,6,16,32,64), ax=axes[1])
    axes[0].set_title("Time against system size for several algoritms")
    axes[1].set_title("Accuracy against system size for the base algorithm")
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
    _, axes = plt.subplots(2,3, figsize=(3*w*2,2*h*2))

    
    axes[0,0].set_title("Smallest - Beam Search")
    axes[0,1].set_title("Full - Beam Search")
    axes[0,2].set_title("Full - ISI")

    ns = (2,4,6,8,10,12,14)
    plot_reg_x_per_z(df, ns, "n", "bs", "smallest", "base_algo", axes[0,0])
    plot_reg_x_per_z(df, ns, "n", "bs", "full", "base_algo", axes[0,1])
    plot_reg_x_per_z(df, ns, "n", "bs", "full", "structured", axes[0,2])

    bss = (2,4,8,16,32,64,128,256)
    plot_reg_x_per_z(df, bss, "bs", "n", "smallest", "base_algo", axes[1,0])
    plot_reg_x_per_z(df, bss, "bs", "n", "full", "base_algo", axes[1,1])
    plot_reg_x_per_z(df, bss, "bs", "n", "full", "structured", axes[1,2])

    for ax in axes.flatten():
        ax.legend(ncols=2)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel("n_calls")
    for ax in axes[0,:]: ax.set_xlabel("beam size")
    for ax in axes[1,:]: ax.set_xlabel("|V|")
    plt.tight_layout()
    plt.savefig(folder+"regressions.pdf")
    plt.show()

def plot_reg_x_per_z(df, zs, z_label, x_label, exh, algo, ax=None, y_label="n_calls"):
    if ax is None: ax = plt.gca()
    for i, z in enumerate(zs):
        index = (
            (df.exh==exh) & (df.algo==algo) & (df[z_label]==z) & df.heuristic.isna()
        )
        
        df_ = df[index].groupby([x_label]).mean([y_label])[[y_label]]
        if not df_.index.size: continue
        X, Y = df_.index.array, df_[y_label].array
        fit, (r2, Y_pred, coefs) = find_model(X,Y)
        ax.plot(X,Y, "x", c=f"C{i}", ls='-', label=f"{z_label}={int(z)}: {fit} {r2:.0%}")
        ax.plot(X,Y_pred, c=f"C{i}",ls='--')

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
    
    index = ((df.model=="base")&(df.n==2)&(df.bs==256)&(df.heuristic.isna())&(df.exh=="full"))
    df_ = df[index].groupby(["algo"]).n_calls.mean()
    print(f"base model, full, n=2, bs=256:            base->ISI: {(df_.loc['base_algo'] - df_.loc['structured'])/df_.loc['base_algo']:.0%} less calls")
    
    index = ((df.model=="base")&(df.n==5)&(df.bs==2)&(df.heuristic.isna())&(df.exh=="full"))
    df_ = df[index].groupby(["algo"]).n_calls.mean()
    print(f"base model, full, n=5, bs=2:              base->ISI: {(df_.loc['base_algo'] - df_.loc['structured'])/df_.loc['base_algo']:.0%} less calls")
    
    index = ((df.model=="non-boolean")&(df.n==10)&(df.bs==16)&(df.heuristic.isna())&(df.exh=="full"))
    df_ = df[index].groupby(["algo"]).n_calls.mean()
    print(f"non-boolean model, full, n=10, bs=16:     base->ISI: {(df_.loc['base_algo'] - df_.loc['structured'])/df_.loc['base_algo']:.0%} less calls")