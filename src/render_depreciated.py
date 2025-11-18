def get_values_det(models, algos, metrics, prefix=""):
    # Initialize recursive default dict
    values = defaultdict( # algos
        lambda: defaultdict( # models
            lambda: defaultdict( # n_attackers
                lambda: defaultdict( # beam_sizes
                    lambda: defaultdict( # contexts
                        lambda: defaultdict(list) # metrics
                    )))))
    
    n_attackers, beam_sizes, contexts = set(), set(), set()
    
    # Convert to values early (assuming Enums or similar)
    algos = sorted([algo.value for algo in algos])
    models = sorted([model.value for model in models])
    contexts = defaultdict(lambda: set())
    
    # Collect data
    for model, algo in tqdm(product(models, algos), desc="Loading data"):
        try:
            data = load_json(f"{prefix}results/{model}-full/{algo}.json")
        except FileNotFoundError: 
            continue
        for datum in data:
            bs, n = datum["beam_size"], datum["n_attacker"]
            beam_sizes.add(bs)
            n_attackers.add(n)
            for res in datum["results"]:
                u = tuple(res["context"])
                contexts[n].add(u)
                for metric in metrics:
                    values[algo][model][n][bs][u][metric].append(res[metric])

    # Sort and create index maps
    n_attackers = sorted(n_attackers)
    beam_sizes = sorted(beam_sizes)
    context_keys = list(range(len(next(iter(contexts.values())))))
    contexts = {key: sorted(value) for key, value in contexts.items()}
    metrics = sorted(metrics)

    # Build keys
    keys = [algos, models, n_attackers, beam_sizes, context_keys, metrics]
    
    # Initialize numpy array
    shape = (len(algos), len(models), len(n_attackers), len(beam_sizes), len(context_keys), len(metrics))
    arr = np.full(shape, np.nan, dtype=float)
    id_value_pairs = [enumerate(key_values) for key_values in keys]
    
    # Fill array
    for key_pairs_list in product(*id_value_pairs):
        algos_pair, models_pair, n_pair, bs_pair, u_pair, m_pair = key_pairs_list
        u_label = contexts[n_pair[1]][u_pair[1]]
        v = values[algos_pair[1]][models_pair[1]][n_pair[1]][bs_pair[1]][u_label][m_pair[1]]
        if v:
            arr[algos_pair[0],models_pair[0],n_pair[0],bs_pair[0],u_pair[0],m_pair[0]] = v[0]
    
    return arr, dict(zip(("algo", "model", "n", "bs", "u", "metric"),keys))

def get_values_stoc(algos, lucb_labels, metrics, prefix=""):
    # Initialize recursive default dict
    values = defaultdict( # algos
        lambda: defaultdict( # lucb_labels
            lambda: defaultdict( # n_attackers
                lambda: defaultdict( # beam_sizes
                    lambda: defaultdict( # contexts
                        lambda: defaultdict( # seeds
                            lambda: defaultdict(list) # metrics
                    ))))))
    
    n_attackers, beam_sizes, contexts, seeds = set(), set(), set(), set()
    
    # Convert to values early (assuming Enums or similar)
    algos = sorted([algo.value for algo in algos])
    contexts = defaultdict(lambda: set())
    
    # Collect data
    for algo, lucb_label in tqdm(product(algos, lucb_labels), desc="Loading data"):
        try:
            data = load_json(f"{prefix}results/noisy-full/{algo}-{lucb_label}.json")
        except FileNotFoundError: 
            print("file not found")
            continue
        for datum in data:
            bs, n = datum["beam_size"], datum["n_attacker"]
            beam_sizes.add(bs)
            n_attackers.add(n)
            for res in datum["results"]:
                u, seed = tuple(res["context"]), res["seed"]
                contexts[n].add(u)
                seeds.add(seed)
                for metric in metrics:
                    values[algo][lucb_label][n][bs][u][seed][metric].append(res[metric])

    # Sort and create index maps
    n_attackers = sorted(n_attackers)
    beam_sizes = sorted(beam_sizes)
    context_keys = list(range(len(next(iter(contexts.values())))))
    contexts = {key: sorted(value) for key, value in contexts.items()}
    seeds = sorted(seeds)
    metrics = sorted(metrics)

    # Build keys
    keys = [algos, lucb_labels, n_attackers, beam_sizes, context_keys, seeds, metrics]
    
    # Initialize numpy array
    shape = (len(algos), len(lucb_labels), len(n_attackers), len(beam_sizes), len(context_keys), len(seeds), len(metrics))
    arr = np.full(shape, np.nan, dtype=float)
    id_value_pairs = [enumerate(key_values) for key_values in keys]
    
    # Fill array
    for key_pairs_list in product(*id_value_pairs):
        algos_pair, lucb_labels_pair, n_pair, bs_pair, u_pair, seed_pair, m_pair = key_pairs_list
        u_label = contexts[n_pair[1]][u_pair[1]]
        v = values[algos_pair[1]][lucb_labels_pair[1]][n_pair[1]][bs_pair[1]][u_label][seed_pair[1]][m_pair[1]]
        if v:
            arr[algos_pair[0],lucb_labels_pair[0],n_pair[0],bs_pair[0],u_pair[0],seed_pair[0],m_pair[0]] = v[0]
    
    return arr, dict(zip(("algo", "lucb_label", "n", "bs", "u", "seed", "metric"),keys))

def extract(arr, keys, **keys_extract):
    key_ids = []
    for key_label, key_values in keys.items():
        key_extract = keys_extract.get(key_label)
        if key_extract is None:
            key_ids.append(slice(None))
        else:
            key_ids.append(key_values.index(key_extract))
    
    return arr[tuple(key_ids)]

def plot_metric(arr, keys, ax=None, c=None, **keys_extract):
    if ax is None: ax= plt.gca()
    ls = "-" if keys_extract["algo"] == AlgoTypes.BASE.value else "--"
    Y = extract(arr, keys, **keys_extract)
    if keys_extract.get("u") is not None:
        if "lucb_label" not in keys_extract:
            ax.plot(keys["bs"], Y, ls=ls, marker="x", c=c)
        else:
            ax.errorbar(keys["bs"], Y.mean(1), yerr=Y.std(1), ls=ls, marker="x", c=c)
    else:
        if "lucb_label" in keys_extract:
            Y = Y.mean(2)
        ax.plot(keys["bs"], Y.mean(1), ls=ls, marker="x", c=c)
    ax.set_xticks(keys["bs"])

