import numpy as np
from tqdm import tqdm
import random as rand
import math
import utilities as uti
from scipy import integrate
import os
import pandas as pd


def normpdf(x, sd, mean):
    a = mean + 1
    var = float(sd)**2
    denom = (2 * math.pi * var)**.5
    num = math.exp(-(float(x) - float(mean))**2 / (2 * var))
    return num / denom


def log_likelihood(D0, D1, N):
    LL0 = D0 @ np.log(N)
    LL0_s = LL0.sum()
    LL1 = np.log(1 - np.exp(D1 @ np.log(N)))
    return LL0_s, LL1


def log_likelihood_update(LL0_s, LL1, N, N_, node, D0, D1):
    # save time by just updating rather than recomputing log likelihood
    # update LL0_s
    LL0_s_new = LL0_s + D0[:, node].sum() * (math.log(N_[node]) -
                                             math.log(N[node]))

    # updat LL1
    LL1_new = LL1.copy()
    for i in range(len(D1)):
        if D1[i, node] == 1:
            LL1_new[i] = np.log(1 - np.exp(D1[i, :] @ np.log(N_)))

    return LL0_s_new, LL1_new


def mcmc(D0, D1, n, iterations, beacons, burn_in=1, record_step=None, sd=1):
    # function to implement MCMC inference on given paths that display RFD (D1)
    # and thos that dont (D0)
    # TODO: speed up (split matrix for RFD and not is odd)

    # initialise (uniform prior)
    N = np.ones((n, 1))
    N = 0.5 * N
    N_ = N.copy()

    LL0_s, LL1 = log_likelihood(D0, D1, N)

    old_likelihood = LL0_s + LL1.sum()
    acceptance = 0
    save = {i: [] for i in range(n)}
    for it in tqdm(range(iterations)):

        #pick random node
        node = -1
        while node < 0 or node in beacons:
            node = rand.choice(range(n))

        # peturb the current state of node
        new = -1
        while (N_[node] + new < 0) or (N_[node] + new) > 1:
            new = np.random.normal(0, sd)

        old = N_[node]
        N_[node] += new

        #get log likelihood updates
        LL0_s_new, LL1_new = log_likelihood_update(LL0_s, LL1, N, N_, node, D0,
                                                   D1)

        # calculate alpha
        new_likelihood = LL0_s_new + LL1_new.sum()
        alpha = new_likelihood - old_likelihood + integrate.quad(
            normpdf, -np.inf, 1, args=(
                sd,
                N_[node],
            )
        )[0] - integrate.quad(normpdf, -np.inf, 1, args=(
            sd,
            old,
        ))[0] + (1 - integrate.quad(normpdf, -np.inf, 0, args=(
            sd,
            N_[node],
        ))[0]) - (1 - integrate.quad(normpdf, -np.inf, 0, args=(
            sd,
            old,
        ))[0])

        # accept or reject move (and update sampels)
        if math.log(rand.random()) < alpha:
            acceptance += 1
            old_likelihood = new_likelihood
            LL0_s = LL0_s_new
            LL1 = LL1_new
            N[node] = N_[node]
            if record_step and it > burn_in and it % record_step == 0:
                for l in range(n):
                    save[l].append(N[l][0])
        else:
            N_[node] = N[node]

    if record_step:
        return save, acceptance

    else:
        return N


def run_mcmc(upd_interval):
    uti.log(f"update interval {upd_interval}min: mcmc starting")
    filename = f"rfd_paths_BeCAUSe_format_{upd_interval}.csv"
    
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

    beacons = set(
        df['path_with_index'].apply(lambda x: node_index[x[-1]]).values)

    # put the data into two matrices D1 are paths with RFD flag and D0 do not have RFD flag
    data_total = np.zeros((number_of_nodes))
    data_nodes0 = np.zeros((number_of_nodes))
    data_nodes1 = np.zeros((number_of_nodes))
    for k, row in df.iterrows():
        for j in row["path_with_index"]:
            data_total[j] += 1
        if row.rfd:
            for j in row["path_with_index"]:
                data_nodes1[j] += 1
        else:
            for j in row["path_with_index"]:
                data_nodes0[j] += 1

    # data is a matrix 1 if node in path, 0 if not
    D0 = np.zeros((len(df.rfd) - sum(df.rfd), number_of_nodes))
    D1 = np.zeros((sum(df.rfd), number_of_nodes))
    k = 0
    j = 0
    for i, row in df.iterrows():
        if row.rfd:
            for nodes in row["path_with_index"]:
                D1[k, nodes] = 1
            k += 1
        else:
            for nodes in row["path_with_index"]:
                D0[j, nodes] = 1
            j += 1

    # implement
    iterations = number_of_nodes * 1000
    save_its = 1000 
    save = np.zeros((number_of_nodes, save_its))
    for k in range(save_its):
        uti.log(f"save iteration {k}/{save_its}")
        N_out = mcmc(D0,
                     D1,
                     D0.shape[1],
                     iterations,
                     beacons,
                     burn_in=number_of_nodes,
                     record_step=None,
                     sd=1)
        save[:, k] = N_out[:, 0]

    mcmc_samples = pd.DataFrame(save)
    mcmc_samples['nodes'] = node_index_inv 
    mcmc_samples = mcmc_samples.set_index('nodes')
    mcmc_samples.to_csv(f"mcmc_samples_{upd_interval}.csv")


# run mcmc for each update interval
upd_intervals = [1,2,3] if "march" in os.getcwd() else [5,10,15]
for upd_interval in upd_intervals:
    run_mcmc(upd_interval)
