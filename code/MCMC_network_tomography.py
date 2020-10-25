import scipy.stats
import networkx as nx
import numpy as np
import random as rand
import time
import math
import matplotlib.pyplot as plt
import collections
from scipy import integrate
import math
import bisect
import sys
import numba
import matplotlib
import os
import pandas as pd

# functionality for parallel computing 
try:
    task = int(os.getenv('task'))
except:
    task = 1

def normpdf(x,sd,mean):
    a = mean+1
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

filepath=''
samplesfilename = 'RFD_mcmc_samples'

## "Read in data and manipulate"
filename = 'rfd_paths_caitlin_format_1.csv'

df = pd.read_csv(filename,sep='|',header=0)
df['path']=df.path.apply(eval)
df['path']=df.path.apply(lambda x: [int(o) for o in x])
nodes = set()
for each in df['path']:
    for i in each:
        nodes.add(i)

i=0
node_index={}
for each in nodes:
    node_index[i] = each
    i+=1
node_index_inv = {v:i for i,v in node_index.items()}

df['Data'] = df.path.apply(lambda x: [node_index_inv[i] for i in x])
df['Data2'] = df.Data.apply(lambda x: [(x[i],x[i+1]) for i in range(len(x)-1)]) # this is for link level inference

firsts = set(df['Data'].apply(lambda x: node_index[x[-1]]).values)

n = len(nodes)


# put the data into two matrices D1 are paths with RFD flag and D0 do not have RFD flag
data_total = np.zeros((n))
data_nodes0 = np.zeros((n))
data_nodes1 =  np.zeros((n))
for k,row in df.iterrows():
    for j in row.Data:
        data_total[j] +=1
    if row.rfd:
        for j in row.Data:
            data_nodes1[j]+=1
    else:
        for j in row.Data:
            data_nodes0[j]+=1
# data is a matrix 1 if node in path, 0 if not

D0  = np.zeros((len(df.rfd)-sum(df.rfd),n))
D1  = np.zeros((sum(df.rfd),n))
k=0
j=0
for i,row in df.iterrows():
    if row.rfd:
        for nodes in row.Data:
            D1[k,nodes] = 1
        k+=1
    else:
        for nodes in row.Data:
            D0[j,nodes] = 1
        j+=1



def log_likelihood(D0,D1,N):
    LL0 = D0 @ np.log(N)
    LL0_s = LL0.sum()
    LL1 = np.log(1-np.exp(D1 @ np.log(N)))
    return LL0_s,LL1

def log_likelihood_update(LL0_s,LL1,N,N_,node,D0,D1):
    # save time by just updating rather than recomputing log likelihood
    # update LL0_s
    LL0_s_new = LL0_s + D0[:,node].sum() * (math.log(N_[node]) - math.log(N[node]))

    # updat LL1
    LL1_new = LL1.copy()
    for i in range(len(D1)):
        if D1[i, node] ==1:
            # print(i)
            LL1_new[i] = np.log(1-np.exp(D1[i,:] @ np.log(N_)))

    return LL0_s_new,LL1_new

def mcmc(D0,D1,n,iterations,burn_in=1,record_step = None,sd=1):
    # function to implement MCMC inference on given paths that display RFD (D1) and thos that dont (D0)
    # TODO: speed up (split matrix for RFD and not is odd)
    
    # initialise (uniform prior)
    N = np.ones((n,1))
    N =  0.5 * N
    N_ = N.copy()

    LL0_s, LL1 = log_likelihood(D0,D1,N)

    old_likelihood = LL0_s + LL1.sum()
    acceptance = 0
    # burn_in = 100
    save = {i:[] for i in range(n)}
    for it in range(iterations):
            #pick random node
            node = -1
            while node <0 or node in firsts:
                node = rand.choice(range(n))
            #node = 1

            # peturb the current state of node 
            new = -1
            while (N_[node]+new < 0) or (N_[node]+new) > 1:
                new = np.random.normal(0,sd)

            
            old = N_[node]
            N_[node] += new

            #get log likelihood updates
            LL0_s_new,LL1_new=log_likelihood_update(LL0_s,LL1,N,N_,node,D0,D1)

            # calculate alpha
            new_likelihood = LL0_s_new+LL1_new.sum()
            alpha = new_likelihood - old_likelihood  + integrate.quad(normpdf,-np.inf,1,args=(sd,N_[node],))[0] \
                                        -integrate.quad(normpdf,-np.inf,1,args=(sd,old,))[0] \
                                        + (1 - integrate.quad(normpdf,-np.inf,0,args=(sd,N_[node],))[0] )\
                                        - (1 - integrate.quad(normpdf,-np.inf,0,args=(sd,old,))[0])

            # accept or reject move (and update sampels)
            if math.log(rand.random())<alpha:
                acceptance+=1
                # print('yes')
            #    print('yes',it)
                old_likelihood= new_likelihood
                LL0_s = LL0_s_new
                LL1 = LL1_new
                N[node] = N_[node]
                if record_step:
                    if it>burn_in:
                        if it%record_step==0:
                            for l in range(n):
                                save[l].append(N[l][0])
            else:
                N_[node] = N[node]

    if record_step:
        return save,acceptance

    else:
        return N


# implement


iterations = n*1000
save_its = 1000 # only do 2 or something if parallelising
save = np.zeros((n,save_its))
for k in range(save_its):
    print(k)
    N_out = mcmc(D0,D1,D0.shape[1],iterations,burn_in = n, record_step=None,sd=1)
    save[:,k] = N_out[:,0]

# if parallelising
#np.save('save/rfd_mcmc_output_iterations='+str(iterations)+'task='+str(task)+'.npy',save)
# else
mcmc_samples = pd.DataFrame(save)
mcmc_samples['nodes']={v:i for i,v in node_index.items()}
mcmc_samples=mcmc_samples.set_index('nodes')
mcmc_samples.to_csv(filepath+filename[:-4]+samplesfilename+'.csv')
