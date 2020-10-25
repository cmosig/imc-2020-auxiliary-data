import numpy as np
import pystan
import pandas as pd
import pickle
import utilities as uti


def run_hmc(upd_interval):
    uti.log(f"{upd_interval}: hmc starting")
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

    # need matrix of D (binary: Dij = 1 if node j on path i)
    # also make vector RFD (binary 1 if RFD else 0)
    number_of_paths = len(df)
    data = np.zeros((number_of_paths, number_of_nodes))
    RFD = np.zeros(number_of_paths, dtype=bool)
    k = 0
    j = len(df) - 1
    for i, row in df.iterrows():
        if row.rfd:
            for nodes in row["path_with_index"]:
                data[k, nodes] = 1
            RFD[k] = 1
            k += 1
        else:
            for nodes in row["path_with_index"]:
                data[j, nodes] = 1
            RFD[j] = 0
            j -= 1

    ## Create the PyStan Model
    stancode = """
    functions {
        real loglikelihoodi_lpdf(real RFD, matrix D, vector P, int n) {
        if (RFD==0){
            return row(D,n) * log(P);
        }
        else{
            return log1m(exp(row(D,n) * log(P)));
        }

        }

        real loglikelihoodRFD0_lpdf(real RFD, matrix D, vector P, int n) {
            return row(D,n) * log(P);
        }
        real loglikelihoodRFD1_lpdf(real RFD, matrix D, vector P, int n) {
            return log1m(exp(row(D,n) * log(P)));
        }

    }
    data {
        int<lower=0> N; //length of paramter vec
        int<lower=0> K; //length of DATA
        matrix[K,N] D; // data of paths (1 if on the path 0 if not)
        int<lower=0, upper=1> RFD[K]; // is the path routeflapdamping
    }
    parameters {
        vector<lower=0, upper=1>[N] P; // vector of parameters
    }

    model {
        P  ~ beta(0.2,0.2);
        for (n in 1:K){
            RFD[n] ~ loglikelihoodi_lpdf(D, P, n);
        }
    }
    """
    # compile the mode
    sm_test = pystan.StanModel(model_code=stancode)
    uti.log(f"{upd_interval}: code compiled")

    # put data in dictionary for PyStan
    data_dict_test = {
        'N': data.shape[1],
        'K': data.shape[0],
        'RFD': RFD.astype(int),
        'D': data
    }

    # decide on required iterations/chains (run with 1000,2 and assess the output)
    iterations = 1000
    chains = 2

    # do the sampling (be patient)
    uti.log(f"{upd_interval}: starting to sample")
    fit_test = sm_test.sampling(data=data_dict_test,
                                iter=iterations,
                                chains=chains)
    uti.log(f"{upd_interval}: sampling done")

    samples = fit_test.to_dataframe().transpose()
    samples = samples.iloc[3:-7]  # get the acutal samples
    samples['nodes'] = {v: i for i, v in node_index.items()}
    samples = samples.set_index('nodes')

    samples.to_csv(f"hmc_samples_{upd_interval}.csv")


# run hmc for each update interval
upd_intervals = [1,2,3] if "march" in os.getcwd() else [5,10,15]
for upd_interval in upd_intervals:
    run_hmc(upd_interval)
