import matplotlib.pyplot as plt
import numpy as np
import gzip
import matplotlib
from matplotlib import rc
import bgpana as bap
from collections import Counter
import seaborn as sns
import itertools

fontsize = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['font.size'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.tick_params(axis='both', labelsize=1)
plt.rcParams['text.usetex'] = True


def transfer_to_links():
    deplexed = []
    # input_file = bap.rsp('./beacon_paths')
    input_file = list(
        map(lambda x: x.decode().split('|'),
            gzip.open('./beacon_paths.gz').read().splitlines()))
    for rc_, path in input_file:
        path = bap.clean_ASpath(path.split(' '))
        links = list(zip(path[:-1], path[1:]))
        for link in links:
            deplexed.append((rc_, link))
    return deplexed


def create_link_heatmap_plot():
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    dataset = transfer_to_links()
    dataset.sort()

    groups = []
    uniquekeys = []
    for k, g in itertools.groupby(dataset, lambda x: x[0]):
        groups.append(set([link for prefix, link in list(g)]))
        uniquekeys.append(k)

    pos_map = {
        "147.28.32.0/24": 3,
        "147.28.36.0/24": 6,
        "147.28.40.0/24": 4,
        "147.28.44.0/24": 2,
        "147.28.48.0/24": 0,
        "147.28.52.0/24": 1,
        "45.132.188.0/24": 5,
    }
    g_k_zip = list(zip(uniquekeys, groups))
    g_k_zip.sort(key=lambda tup: pos_map[tup[0]])
    uniquekeys, groups = zip(*g_k_zip)

    group_dict = dict(zip(uniquekeys, groups))
    combs = list(itertools.product(uniquekeys, repeat=2))
    # print("combs", combs)
    plot_data = list(
        map(
            lambda g: len(group_dict[g[1]].intersection(group_dict[g[0]])) /
            len(group_dict[g[1]]), combs))
    plot_data = np.array_split(plot_data, 7)
    xticklabels = [
        "Japan", "USA", "Thailand", "S. Africa", "Brasil", "Germany", "Denmark"
    ]
    sns.heatmap(
        data=plot_data,
        ax=ax,
        xticklabels=xticklabels,
        yticklabels=xticklabels,
        cmap="GnBu",
        cbar_kws={'label': 'Link Similarity = $\\frac{|Y \cap X|}{|Y|}$'})
    # vmin=0.4,
    # vmax=1)
    ax.set_aspect(1)

    fig.savefig("beacon_link_heatmap.pdf", bbox_inches="tight")


create_link_heatmap_plot()
