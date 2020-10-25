import numpy as np
import os
import utilities as uti
from collections import Counter
import pandas as pd
import pymc3  # only need this for the HPD calc.  if too hard just use meanss


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


def create_results_df_nodeind(node_index, hmc_samples=None, mcmc_samples=None):
    # function creates a dataframe of the risks using the functions above
    # hmc_samples is a df of samples as read above
    # mcmc_samples is a df of samples from mcmc
    results = pd.DataFrame({"nodes": node_index})
    if mcmc_samples is not None:
        results['ave_mcmc'] = mcmc_samples.mean(axis=1).values
        results['risk_mcmc_mean'] = results['ave_mcmc'].apply(risk_level_mean)
        results['credible_interval_mcmc'] = results.nodes.apply(
            lambda i: pymc3.stats.hpd(np.array(mcmc_samples.loc[i])))
        results['risk_mcmc_hpd'] = results['credible_interval_mcmc'].apply(
            risk_level_hpd)

    if hmc_samples is not None:
        results['ave_hmc'] = hmc_samples.mean(axis=1).values
        results['risk_hmc_mean'] = results['ave_hmc'].apply(risk_level_mean)
        results['credible_interval_hmc'] = results.nodes.apply(
            lambda i: pymc3.stats.hpd(np.array(hmc_samples.loc[i])))
        results['risk_hmc_hpd'] = results['credible_interval_hmc'].apply(
            risk_level_hpd)

    results['total_risk'] = results.apply(total_risk, axis=1)
    return results


def create_summary(upd_interval):
    filename = f"rfd_paths_BeCAUSe_format_{upd_interval}.csv"

    uti.log(f"update interval {upd_interval}min: creating summary")
    # Read in data and manipulate
    df = pd.read_csv(filename,
                     sep='|',
                     usecols=[2, 3],
                     converters={"path": lambda x: [int(o) for o in eval(x)]})

    # get unique nodes and create node index
    node_index = df["path"].explode().drop_duplicates().sort_values().to_list()
    node_index_inv = dict(zip(node_index, range(len(node_index))))

    # translate nodes on path to their index
    df['path_with_index'] = df.path.apply(
        lambda x: [node_index_inv[i] for i in x])
    number_of_nodes = len(node_index)

    # load mcmc and hmc samples
    mcmc_samples = pd.read_csv(f'mcmc_samples_{upd_interval}.csv.gz',
                               index_col=0)
    hmc_samples = pd.read_csv(f'hmc_samples_{upd_interval}.csv.gz',
                              index_col=0)

    results = create_results_df_nodeind(node_index, hmc_samples, mcmc_samples)
    results.set_index('nodes', inplace=True)

    flags2 = set()

    uti.log(f"update interval {upd_interval}min: finding extra flags")
    for w, each_path in enumerate(df[df.rfd].path.values):
        if max(results.loc[p].total_risk for p in each_path) <= 3:
            path_means = np.array(
                [np.mean(hmc_samples.loc[p]) for p in each_path])
            # pick the one with the highest prob of being the min
            c_sort = sorted(Counter(hmc_samples.loc[each_path].apply(
                lambda x: np.argmin(x))).items(),
                            key=lambda kv: kv[1],
                            reverse=True)
            if c_sort[0][1] / len(hmc_samples.iloc[0]) > 0.8:
                flags2.add(c_sort[0][0])

    for each in flags2:
        results.loc[each, 'total_risk'] = 4

    results.to_csv(f"summaries_{upd_interval}.csv")


# create summaries for each update interval
upd_intervals = [1, 2, 3] if "march" in os.getcwd() else [5, 10, 15]
for upd_interval in upd_intervals:
    create_summary(upd_interval)
