import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import itertools
from matplotlib_venn import venn3
import bgpana as bap
from collections import Counter

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
    input_file = bap.rsp('./projects_paths')
    for rc_, path in input_file:
        path = bap.clean_ASpath(path.split(' '))
        links = list(zip(path[:-1], path[1:]))
        for link in links:
            deplexed.append((rc_, link))
    return deplexed


def transfer_to_ases():
    deplexed = []
    input_file = bap.rsp('./projects_paths')
    for rc_, path in input_file:
        path = bap.clean_ASpath(path.split(' '))
        for asn in path:
            deplexed.append((rc_, asn))
    return deplexed


datasets = [transfer_to_links(), transfer_to_ases()]
filenames = ['route_collector_path_links.pdf', 'route_collector_path_ases.pdf']

for projects_vps, filename in zip(datasets, filenames):

    # sort by IP
    projects_vps.sort(key=lambda x: x[0])

    shares = Counter([x[0] for x in projects_vps])
    project_names, vp_counts = tuple(zip(*shares.items()))

    fig, ax = plt.subplots(figsize=(3.3 * 0.6, 3.5))

    correct_names = {
        "routeviews": "Routeviews",
        "Isolario": "Isolario",
        "ris": "RIPE RIS"
    }

    groups = []
    uniquekeys = []
    for k, g in itertools.groupby(projects_vps, lambda x: x[0]):
        groups.append(list(g))
        uniquekeys.append(k)
    project_names = list(map(lambda name: correct_names[name], uniquekeys))

    out = venn3(list(map(lambda y: set([x[1] for x in y]), groups)),
                project_names)
    for text in out.subset_labels:
        text.set_fontsize(fontsize - 2)
    for text in out.set_labels:
        text.set_fontsize(fontsize)

    out.get_label_by_id("B").set_position((-0.1, 0.6))
    out.get_label_by_id("A").set_position((-0.17, 0.52))

    plt.savefig(filename, bbox_inches='tight')
