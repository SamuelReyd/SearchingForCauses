import pandas as pd
from evaluation import evaluate_full
from benchmark_models import *


sanity_checks = (
    ("Forest Fire Disjunctive", scm_ff_disj, exp_ff_disj),
    ("Forest Fire Conjunctive", scm_ff_conj, exp_ff_conj),
    ("Rock Throwing", scm_suzzy, exp_suzzy),
    ("Prisoners", scm_prisoners, exp_prisoners),
    ("Assassin Variant 1", scm_assassin_1, exp_assassin_1),
    ("Assassin Variant 2", scm_assassin_2, exp_assassin_2),
    ("Lamp", scm_lamp, exp_lamp),
    ("Forest Fire Extended", scm_ff_ext, exp_ff_ext),
    ("Ranch", scm_ranch, exp_ranch),
    ("Ranch Extended", scm_ranch_ext, exp_ranch_ext),
    ("Vote", scm_vote, exp_vote),
    ("Vote 3 Ways", scm_vote_ext, exp_vote_ext),
    # ("Railroad no flip", scm_railroad_no_flip, exp_railroad_no_flip),
    # ("Railroad with flip", scm_railroad_flip, exp_railroad_flip),
    ("Rock Throwing Extended",scm_rock_thr_ext, exp_suzzy_ext),
)

def run_sanity_checks(sanity_checks, verbose=True):
    lines = []
    index = []
    for label, model, exp_causes in sanity_checks:
        index.append([label, str(exp_causes).replace("{", r"\{").replace("}",r"\}")])
        exp_causes = [tuple(cause) for cause in exp_causes]
        model.find_causes(ISI=True)
        if verbose: 
            print(label, ":", model.causes)
    
        calls = [model.n_calls]
        times = [f"{model.identification_time*1000:.2f}"]
        out = [model.causes_hashable]
        model.find_causes(ISI=False, beam_size=-1, early_stop=False, max_steps=-1)
        calls += [model.n_calls]
        times += [f"{model.identification_time*1000:.2f}"]
        out += [model.causes_hashable]
        
        line = calls + times + [r"{\color{green}\checkmark}" if evaluate_full(causes, exp_causes)["dice"] > .99 else r"{\color{red}\text{\sffamily X}}" for causes in out]
        lines.append(line)
    return lines, index

def plot_sanity_checks(lines, index, show_pd = False):
    columns=pd.MultiIndex.from_tuples([
        ('n_calls', "MBS"),
        ('n_calls', "ISI"),
        ('t (ms)', "MBS"),
        ('t (ms)', "ISI"),
        ('correct?', "MBS"),
        ('correct?', "ISI"),
    ])
    df = pd.DataFrame(lines, columns=columns, index=pd.MultiIndex.from_tuples(index, names=["Model","Expected causes"]))

    if show_pd:
        df[('correct?', "MBS")] = df[('correct?', "MBS")].apply(lambda x: "V" if x == "{\color{green}\checkmark}" else "X")
        df[('correct?', "ISI")] = df[('correct?', "ISI")].apply(lambda x: "V" if x == "{\color{green}\checkmark}" else "X")
        print(df)
    else:
        print(df.to_latex().replace("_", r"\_").replace("'", ""))