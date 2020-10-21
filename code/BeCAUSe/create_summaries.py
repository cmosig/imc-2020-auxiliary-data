import scipy.stats
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
import collections
import math
import matplotlib
# import pystan
import matplotlib
from scipy.stats import gaussian_kde
import pandas as pd
import pymc3  # only need this for the HPD calc.  if too hard just use meanss

# to create the results df from the mcmc_samples and hmc_samples


# change the values in these functions if you like
def total_risk(x):
    try:
        tr = max([
            x.risk_mcmc_hpd, x.risk_hmc_mean, x.risk_mcmc_hpd, x.risk_hmc_hpd
        ])
    except:
        try:
            tr = max([x.risk_mcmc_mean, x.risk_mcmc_hpd])
        except:
            tr = max([x.risk_hmc_mean, x.risk_hmc_hpd])

    if tr == 3:
        try:
            if x.ave_hmc > 0.9:
                return 2
            elif x.ave_hmc < 0.1:
                return 4
            else:
                return 3
        except AttributeError:
            return 3
    else:
        return tr


def risk_level_mean(x):
    if x > 0.85:
        return 1
    elif x > 0.6:
        return 2
    elif x > 0.4:
        return 3
    elif x > 0.15:
        return 4
    else:
        return 5


def risk_level_hpd(x):

    if x[1] < 0.15:  # bad
        return 5
    elif x[0] > 0.85:
        return 1

    elif x[1] < 0.4:
        return 4
    elif x[0] > 0.6:
        return 2

    else:
        return 3


# create the summaries of the distributions given the node index dictionary and
# the samples from the HMC and MCMC


def create_results_df_nodeind(node_index, hmc_samples=None, mcmc_samples=None):
    # function creates a dataframe of the risks using the functions above
    # hmc_samples is a df of samples as read from output of HMC_network_tomography.py
    # mcmc_samples is a df of samples from mcmc from MCMC_network_tomography.py

    results = pd.DataFrame(list(node_index.values()))
    results.columns = ['nodes']

    if mcmc_samples is not None:
        results['ave_mcmc'] = mcmc_samples.mean(axis=1).values
        results['risk_mcmc_mean'] = results['ave_mcmc'].apply(risk_level_mean)
        results['credible_interval_mcmc'] = results.nodes.apply(
            lambda i: pymc3.stats.hpd(mcmc_samples.loc[i]))
        results['risk_mcmc_hpd'] = results['credible_interval_mcmc'].apply(
            risk_level_hpd)

    if hmc_samples is not None:
        results['ave_hmc'] = hmc_samples.mean(axis=1).values
        results['risk_hmc_mean'] = results['ave_hmc'].apply(risk_level_mean)
        results['credible_interval_hmc'] = results.nodes.apply(
            lambda i: pymc3.stats.hpd(hmc_samples.loc[i]))
        results['risk_hmc_hpd'] = results['credible_interval_hmc'].apply(
            risk_level_hpd)

    results['total_risk'] = results.apply(total_risk, axis=1)
    return results


datasets = [
    "rfd_paths_BeCAUSe_format_1", "rfd_paths_BeCAUSe_format_2",
    "rfd_paths_BeCAUSe_format_3", "rfd_paths_BeCAUSe_format_5",
    "rfd_paths_BeCAUSe_format_10", "rfd_paths_BeCAUSe_format_15"
]

dataset = datasets[1]
datatype = "RFD"

df = pd.read_csv(dataset + '.csv', sep='|')
df['path'] = df.path.apply(eval)
df['path'] = df.path.apply(lambda x: [int(o) for o in x])
nodes = set()
for each in df['path']:
    for i in each:
        nodes.add(i)

# create node-index mapping
node_index_inv = dict(zip(nodes, range(len(nodes))))
node_index = nodes
node_count = len(nodes)

# Data is the index on the paths
df['Data'] = df["path"].apply(lambda x: [node_index_inv[i] for i in x])

# Read in MCMC outputs (if done in parallel with a single sample as each
# output) - change filenames as required
try:  # test if the combined samples already exists
    mcmc_samples = pd.read_csv('results/' + dataset + 'RFD_mcmc_samples.csv',
                               index_col=0)
    print('found mcmc samples ')
except FileNotFoundError:  # if not combine them
    print('not found, finding')
    S = 1000  # number of samples taken (hmc = 2000, mcmc = 1000)
    phoenix_mat = np.zeros((node_count, S * 2))
    i = 0
    not_found = 0
    for task in range(1000):
        try:
            mat_read = np.load('results/raw/' + dataset +
                               '_mcmc_output_task=' + str(task) + '.npy')
            phoenix_mat[:, i:i + 2] = mat_read
            i += 2
        except FileNotFoundError:
            not_found += 1
            print('notfound', task)

    mcmc_samples = pd.DataFrame(phoenix_mat[:, :(S - not_found) * 2])
    mcmc_samples['nodes'] = node_index_inv
    mcmc_samples = mcmc_samples.set_index('nodes')
    mcmc_samples.to_csv('results/' + dataset + '_mcmc_samples.csv')

# Read in the HMC samples (if not found here then we will need to change the
# index from the PyStan output)
try:
    hmc_samples = pd.read_csv('results/' + dataset + 'RFD_hmc_samples.csv',
                              index_col=0)
    print('found hmc')
except FileNotFoundError:
    print('finding hmc samples')
    hmc_samples = pd.read_csv('results/raw/' + dataset + datatype +
                              '_hmc_samples.csv',
                              index_col=0)
    hmc_samples = hmc_samples.reset_index(drop=True)
    hmc_samples['nodes'] = node_index_inv
    hmc_samples = hmc_samples.set_index('nodes')
    hmc_samples.to_csv('results/' + dataset + datatype + '_hmc_samples.csv')

# Now create the results DF that will give a category based on the functions
# above.
results = create_results_df_nodeind(node_index, hmc_samples, mcmc_samples)

# If there is no flagged node on the path then we find the most likely RFD node
flags2 = set()
no_flag = 0
print('finding extra flags')
for w, each_path in enumerate(df[df.rfd].Data.values):
    if max(results.iloc[p].total_risk for p in each_path) <= 3:
        path_means = np.array(
            [np.mean(hmc_samples.iloc[p]) for p in each_path])
        # pick the one with the highest prob of being the min
        c_sort = sorted(Counter(
            hmc_samples.iloc[each_path].apply(lambda x: np.argmin(x))).items(),
                        key=lambda kv: kv[1],
                        reverse=True)
        if c_sort[0][1] / len(hmc_samples.iloc[0]) > 0.8:
            flags2.add(each_path[c_sort[0][0]])
            results.loc[each_path[c_sort[0][0]], 'total_risk'] = 4
        elif c_sort[0][1] / len(hmc_samples.iloc[0]) > 0.5:
            results.loc[each_path[c_sort[0][0]], 'total_risk'] = 3

for each in flags2:
    results.loc[each, 'total_risk'] = 4

results = results.set_index('nodes')
results.to_csv('results/summaries/summaries_with_flags' + dataset + datatype +
               '.csv')

# PLOTTING
col = 9
row = 13
fig, ax = plt.subplots(row, col, figsize=(10, 25))
k = list(range(col)) * row
for i, node in enumerate(results[results.total_risk > 2].nodes.values):
    # ax[i//col][k[i]].hist(samples.iloc[node],bins =np.linspace(0,1,11),color = 'k',density=True ,alpha =0.2)
    ax[i // col][k[i]].fill_between(
        np.linspace(0, 1, 100), 0,
        kde_scipy(hmc_samples.loc[node], np.linspace(0, 1, 100)))
    ax[i // col][k[i]].set_xlim([0, 1])
    ax[i // col][k[i]].set_yticks([])
    ax[i // col][k[i]].set_xticks([0, 1])
    #ax[i//col][k[i]].set_title('AS:' + str(node_index[node]))
    ax[i // col][k[i]].set_title(str(node))
ax[-1, -1].set_yticks([])
ax[-1, -1].set_xticks([])
fig.subplots_adjust(left=0.125, right=0.9, bottom=0.05, top=0.95, hspace=1.5)
