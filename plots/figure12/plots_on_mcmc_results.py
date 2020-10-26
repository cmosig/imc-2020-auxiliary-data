import matplotlib.pyplot as plt
import pandas as pd

fontsize = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['font.size'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.tick_params(axis='both', labelsize=1)
plt.rcParams['text.usetex'] = True

upd_intervals = [1, 2, 3, 5, 10, 15]

normal = []
thresh_only = []

for upd_int in upd_intervals:
    mcmc_extra_flags = pd.read_csv(
        f'summaries_{upd_int}_extra_flags.csv'
    ).set_index("nodes")
    normal.append(mcmc_extra_flags["total_risk"] >= 4)

    # pinpointing purely based on threshold without step 2
    mcmc_thresholds_only = pd.read_csv(
        f'summaries_{upd_int}_thresholds_only.csv'
    ).set_index("nodes")
    thresh_only.append(mcmc_thresholds_only["total_risk"] >= 4)

# merge series
normal = pd.concat(normal, axis=1).dropna()
normal.columns = upd_intervals
thresh_only = pd.concat(thresh_only, axis=1).dropna()
thresh_only.columns = upd_intervals

# get total ases
total_count = normal.shape[0]
print("total_count", total_count)

assert normal.shape[0] == thresh_only.shape[0]

# count rfd ASes
normal = normal.apply(lambda x: len(x[x]))
thresh_only = thresh_only.apply(lambda x: len(x[x]))


def num_rfd_nodes():
    fig, ax = plt.subplots(figsize=(3.3, 2))

    ax.bar(range(len(normal)), (normal / total_count) * 100,
           color="steelblue",
           width=0.6,
           label="Inconsistent")
    ax.bar(range(len(thresh_only)), (thresh_only / total_count) * 100,
           color="orange",
           width=0.6,
           label="Consistent")
    ax.set_ylabel("RFD ASs [\%]")
    ax.set_xlabel("Update Interval [Minutes]")
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels([1, 2, 3, 5, 10, 15])
    ax.legend(fontsize=9)

    fig.savefig("deployment.pdf", bbox_inches="tight")
    plt.clf()


num_rfd_nodes()
