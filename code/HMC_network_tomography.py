import numpy as np
import random as rand
import math
import pystan
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pickle
import pandas as pd

samples_filename = 'RFD_hmc_samples.csv'

## "Read in data and manipulate"
dataset = "rfd_paths_BeCAUSe_format_1.csv"
df = pd.read_csv(dataset, sep='|', usecols=[1,2,3,4])
df['path'] = df["path"].apply(eval)
df['path'] = df["path"].apply(lambda x: [int(o) for o in x])
nodes = set()
for each in df['path']:
    for i in each:
        nodes.add(i)
i = 0
node_index = {}
for each in nodes:
    node_index[i] = each
    i += 1
node_index_inv = {v: i for i, v in node_index.items()}
df['Data'] = df.path.apply(lambda x: [node_index_inv[i] for i in x])
n = len(nodes)

# need matrix of D (binary: Dij = 1 if node j on path i)
# also make vector RFD (binary 1 if RFD else 0)
D = np.zeros((len(df), n))
RFD = np.zeros(len(df), dtype=bool)
k = 0
j = len(df) - 1
for i, row in df.iterrows():
    if row.rfd:
        for nodes in row.Data:
            D[k, nodes] = 1
        RFD[k] = 1
        k += 1
    else:
        for nodes in row.Data:
            D[j, nodes] = 1
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

# save if required
# with open('rfd_basic_model.pkl', 'wb') as f:
#     pickle.dump(sm_test, f)

# put data in dictionary for PyStan
data = D

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
fit_test = sm_test.sampling(data=data_dict_test,
                            iter=iterations,
                            chains=chains)  #,algorithm="HMC")
# print(fit_test)

samples = fit_test.to_dataframe().transpose()
samples = samples.iloc[3:-7]  # get the acutal samples

samples.to_csv(samples_filename)

with open('rfd_basic_model_fit.pkl', 'wb') as f:
    pickle.dump(fit_test, f)

# Stan code for different Prior distributions

# UNIFORM prior
stancode_uniform_prior = """
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
    for (n in 1:K){
        RFD[n] ~ loglikelihoodi_lpdf(D, P, n);
    }
}
"""
sm_uniform = pystan.StanModel(model_code=stancode_uniform_prior)

iterations = 2000
chains = 2

fit_uniform = sm_uniform.sampling(data=data_dict_test,
                                  iter=iterations,
                                  chains=chains)  #,algorithm="HMC")

samples = fit_uniform.to_dataframe().transpose()
samples = samples.iloc[3:-7]

# samples.to_csv('RFD_hmc_samples_priorU.csv')

# with open('rfd_basic_model_priorU_fit.pkl', 'wb') as f:
#     pickle.dump(fit_uniform, f)

# prior with weight at 0 (beta distribution)
stancode_prior0 = """
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
    P  ~ beta(1,3);
    for (n in 1:K){
        RFD[n] ~ loglikelihoodi_lpdf(D, P, n);
    }
}
"""
sm_prior0 = pystan.StanModel(model_code=stancode_prior0)

iterations = 2000
chains = 2

fit_prior0 = sm_prior0.sampling(data=data_dict_test,
                                iter=iterations,
                                chains=chains)  #,algorithm="HMC")
# print(fit_test)

samples = fit_prior0.to_dataframe().transpose()
samples = samples.iloc[3:-7]

# samples.to_csv('RFD_hmc_samples_prior0.csv')
#
# with open('rfd_basic_model_prior0_fit.pkl', 'wb') as f:
#     pickle.dump(fit_prior0, f)

# prior with weight at 1
stancode_prior1 = """
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
    P  ~ beta(3,1);
    for (n in 1:K){
        RFD[n] ~ loglikelihoodi_lpdf(D, P, n);
    }
}
"""
sm_prior1 = pystan.StanModel(model_code=stancode_prior1)

iterations = 2000
chains = 2

fit_prior1 = sm_prior1.sampling(data=data_dict_test,
                                iter=iterations,
                                chains=chains)  #,algorithm="HMC")
# print(fit_test)

samples = fit_prior1.to_dataframe().transpose()
samples = samples.iloc[3:-7]

# samples.to_csv('RFD_hmc_samples_prior1.csv')
#
# with open('rfd_basic_model_prior1_fit.pkl', 'wb') as f:
#     pickle.dump(fit_prior1, f)

# plotting if you like


def kde_scipy(x, x_grid, bandwidth=None, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)
    return kde.evaluate(x_grid)


row = 21
col = 29

fig, ax = plt.subplots(row, col, figsize=(30, 40))
k = list(range(col)) * row
hpd = True
mean = False
for i in range(n):
    ax[i // col][k[i]].hist(samples.iloc[i],
                            bins=np.linspace(0, 1, 11),
                            color='k')
    ax[i // col][k[i]].plot(samples.iloc[i], np.linspace(0, 1, 100), color='b')

    ax[i // col][k[i]].set_xlim([0, 1])
    ax[i // col][k[i]].set_yticks([])
    ax[i // col][k[i]].set_xticks([])
