#CODE TO READ IN THE MCMC SAMPLES AND HMC SAMPLES TO CREATE _SAMPLES.CSV

N = #number of nodes

filepathsamples = 'results/sample_results'
filepath = 'results/'

# NEED NODE INDEX
try:
    with open(filepath + dataset + 'node_index.pkl', 'rb') as f:
        node_index = pickle.load(f)
except:
    print('no node index assume identity')
    node_index = {i: i for i in range(N)}

S = 1000  # number of samples taken (hmc = 2000, mcmc = 1000)
K = 2  # SAMPLES PER FILE
phoenix_mat = np.zeros((n, S * K))
i = 0
not_found = 0
for task in range(S):
    try:
        print(task)
        mat_read = np.load(filepathsamples + dataset + '_mcmc_output_task=' +
                           str(task) + '.npy')
        phoenix_mat[:, i:i + K] = mat_read
        i += K
    except FileNotFoundError:
        not_found += 1
        print('notfound', task)

mcmc_samples = pd.DataFrame(phoenix_mat[:, :(S - not_found) * 2])
mcmc_samples['nodes'] = {v: i for i, v in node_index.items()}
mcmc_samples = mcmc_samples.set_index('nodes')
mcmc_samples.to_csv(filepath + dataset + '_mcmc_samples.csv')

# IF YOU NEED TO CHANGE THE INDEX OF HMC SAMPLES
print('finding hmc samples')
hmc_samples = pd.read_csv(filepathsamples + dataset + '_hmc_samples.csv',
                          index_col=0)
hmc_samples = hmc_samples.reset_index(drop=True)
hmc_samples['nodes'] = {v: i for i, v in node_index.items()}
hmc_samples = hmc_samples.set_index('nodes')
hmc_samples.to_csv(filepath + dataset + datatype + '_hmc_samples.csv')
