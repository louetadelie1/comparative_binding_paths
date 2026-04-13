import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
import math
import pickle
import itertools
import pickle
import itertools
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import smplotlib
from matplotlib.colors import LinearSegmentedColormap, Normalize

def cleanup(d):
    return {tuple(int(k) for k in key): float(value[0]) for key, value in d.items()}


def enrichment_factor(source_rank, target_rank, k):
    actives = source_rank[:k]
    topk_target = target_rank[:k]
    H = k
    N = len(target_rank)
    hits = len(set(actives) & set(topk_target))
    return (hits / k) / (H / N)

paths = [[
    "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_ph5_278k/pickled_files/p_eq_keys_trip.pckl",
    # "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_ph5_278_v2/pickled_files/p_eq_keys_trip.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_ph5_298k/pickled_files/p_eq_keys_trip.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_ph7_278k/pickled_files/p_eq_keys_trip.pckl"],

    ["/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_D4/pickled_files/p_eq_keyss_triplet.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_D8/pickled_files/p_eq_keyss_triplet.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/abeta_g5_new_protocol/pickled_files/p_eq_keyss_triplet.pckl"],

    ["/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_50/pickled_files/p_eq_keyss_triplet.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_40/pickled_files/p_eq_keyss_triplet.pckl",
    # "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_12/pickled_files/p_eq_keyss_triplet.pckl",
    # "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_20/pickled_files/p_eq_keyss_triplet.pckl",
    # "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_26/pickled_files/p_eq_keyss_triplet.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_30/pickled_files/p_eq_keyss_triplet.pckl"],
    # "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_4/pickled_files/p_eq_keyss_triplet.pckl"],

    ["/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_cm8/pickled_files/p_eq_keyss_triplet.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_cm10/pickled_files/p_eq_keyss_triplet.pckl",
    "/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/medin_urea/pickled_files/p_eq_keyss_triplet.pckl"],
]

label_map = {
    "alpha_syn_lig_40": "Lig 40",
    "alpha_syn_lig_20": "Lig 20",
    "alpha_syn_lig_4":  "Lig 4",
    "alpha_syn_lig_26": "Lig 26",
    "alpha_syn_lig_30": "Lig 30",
    "alpha_syn_lig_12": "Lig 12",
    "alpha_syn_lig_50": "Lig 50",
    "medin_cm8": "CM8",
    "medin_cm10": "CM10",
    "medin_urea": "Urea",
    "abeta_D4": "D4",
    "abeta_D8": "D8",
    "abeta_g5_new_protocol": "G5",
    "abeta_ph5_278k": "pH5 278k",
    "abeta_ph5_298k": "pH5 298k",
    "abeta_ph7_278k": "pH7 278k",
}


def get_cmap(inp):
    if "_ph" in inp.split('/')[-3]: 
        return LinearSegmentedColormap.from_list(
            "abeta_blue", ["#edf0fa", "#2f3f7f"]
        )
    elif "medin" in inp.split('/')[-3]:
        return LinearSegmentedColormap.from_list(
            "medin_green", ["#edf8f2", "#5a8f7b"]
        )
    elif "alpha_syn" in inp.split('/')[-3]:
        return LinearSegmentedColormap.from_list(
            "alpha_purple", ["#e0d4f7", "#5e3c99"]
        )
    else:
        return LinearSegmentedColormap.from_list(
            "abeta_blue_2", ["#eef4fa", "#355f8c"])
    

for system in paths:
# palette = sns.color_palette(["#edf0fa", "#2f3f7f"])
    matrices = []
    for path in system:
        with open(path, "rb") as f:
            data = pickle.load(f)
        cleaned = {
            tuple(int(k) for k in key): float(val[0])
            for key, val in data.items()
        }

        cleaned_sorted = dict(
            sorted(cleaned.items(), key=lambda x: x[1], reverse=True)
        )

        matrices.append(cleaned_sorted)

    ranked_lists = [list(m.keys()) for m in matrices]
    #pairs = list(itertools.permutations(range(len(ranked_lists)), 2))
    pairs = list(itertools.combinations(range(len(ranked_lists)), 2))

    # ==================================================
    # PART 1 — ENRICHMENT FACTOR BAR PLOTS (ONE FIGURE)
    # ==================================================
    cutoffs = [20, 50, 100, 200, 400]

    # cols = 2
    # rows = math.ceil(len(pairs) / cols)
    # fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    # axes = axes.flatten()

    n_panels = len(pairs)
    ncols = 3
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        #figsize=(4.2 * ncols, 3.2 * nrows),
        figsize=(8 * ncols, 5 * nrows),

        sharey=True
    )
    axes = np.array(axes).reshape(-1)
    width = 0.35

    # for ax, lig in zip(axes, pairs):

    for idx, (i, j) in enumerate(pairs):

        ranked_A = ranked_lists[i]
        ranked_B = ranked_lists[j]

        ef_AtoB = []
        ef_BtoA = []

        for k in cutoffs:
            actives_A = ranked_A[:k]
            actives_B = ranked_B[:k]
            N = len(ranked_B)

            hits_AB = len(set(actives_A) & set(actives_B))
            ef_AtoB.append((hits_AB / k) / (k / N))

            hits_BA = len(set(actives_B) & set(actives_A))
            ef_BtoA.append((hits_BA / k) / (k / N))

        ax = axes[idx]
        x = np.arange(len(cutoffs))
        system_names = [p.split('/')[-3] for p in system]

        ax.bar(
            x - width / 2,
            ef_AtoB,
            width,
            color=get_cmap(path)(0.5),
            label=f"{label_map.get(system_names[i], system_names[i])} → "
                f"{label_map.get(system_names[j], system_names[j])}"
        )

        ax.bar(
            x + width / 2,
            ef_BtoA,
            width,
            color=get_cmap(path)(0.75),
            label=f"{label_map.get(system_names[j], system_names[j])} → "
                f"{label_map.get(system_names[i], system_names[i])}"
        )

        ax.set_xticks(x)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_xticklabels(cutoffs)
        ax.set_ylabel("Enrichment Factor")
        ax.set_xlabel("Top-k cutoff")
        ax.set_title(
            f"{label_map.get(system_names[i], system_names[i])} vs "
            f"{label_map.get(system_names[j], system_names[j])}",fontsize=20
        )
        ax.legend(fontsize=15)
        ax.grid(alpha=0.3)

        if idx % ncols != 0:   # not first column
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    for ax in axes[len(pairs):]:
        ax.axis("off")

    plt.suptitle("Pairwise Enrichment Factors (All Systems)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # ==================================================
    # PART 2 — AUC ENRICHMENT CURVES (ONE FIGURE)
    # ==================================================
    TOP_ACTIVES = 30

    # cols = 2
    # rows = math.ceil(len(pairs) / cols)
    # fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    # axes = axes.flatten()
    # n_panels = len(system)
    # ncols = 4
    # nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        #figsize=(4.2 * ncols, 3.2 * nrows),
        figsize=(8 * ncols, 5 * nrows),

        sharey=True
    )
    axes = np.array(axes).reshape(-1)
    width = 0.35
    for idx, (i, j) in enumerate(pairs):

        ranked_A = ranked_lists[i]
        ranked_B = ranked_lists[j]

        actives_A = set(ranked_A[:TOP_ACTIVES])
        actives_B = set(ranked_B[:TOP_ACTIVES])

        N = len(ranked_B)

        hits_AB = 0
        frac_scan_AB = []
        frac_find_AB = []

        for n, item in enumerate(ranked_B):
            if item in actives_A:
                hits_AB += 1
            frac_scan_AB.append((n + 1) / N)
            frac_find_AB.append(hits_AB / TOP_ACTIVES)

        auc_AB = np.trapz(frac_find_AB, frac_scan_AB)

        hits_BA = 0
        frac_scan_BA = []
        frac_find_BA = []

        for n, item in enumerate(ranked_A):
            if item in actives_B:
                hits_BA += 1
            frac_scan_BA.append((n + 1) / N)
            frac_find_BA.append(hits_BA / TOP_ACTIVES)

        auc_BA = np.trapz(frac_find_BA, frac_scan_BA)

        ax = axes[idx]

        ax.plot(
            frac_scan_AB,
            frac_find_AB,
            color=get_cmap(path)(0.5),
            linewidth=2,
            label=f"{label_map.get(system_names[i], system_names[i])} → "
                f"{label_map.get(system_names[j], system_names[j])}"
        )

        ax.plot(
            frac_scan_BA,
            frac_find_BA,
            color=get_cmap(path)(0.75),
            linestyle="--",
            linewidth=2,
            label=f"{label_map.get(system_names[j], system_names[j])} → "
                f"{label_map.get(system_names[i], system_names[i])}"
        )

        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_xlabel("Fraction of library scanned",fontsize=20)
        ax.set_ylabel(f"Fraction of top {TOP_ACTIVES} actives found",fontsize=20)
        ax.set_title(
            f"{label_map.get(system_names[i], system_names[i])} vs "
            f"{label_map.get(system_names[j], system_names[j])}",fontsize=20
        )
        ax.legend(fontsize=15)
        ax.grid(alpha=0.3)

        if idx % ncols != 0:   
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    for ax in axes[len(pairs):]:
        ax.axis("off")

    plt.suptitle(f"AUC Enrichment Curves (TOP_ACTIVES={TOP_ACTIVES})", fontsize=18)
    plt.tight_layout()
    plt.show()
    