import pandas as pd
import pymc3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = "newrfd_paths_caitlin_format_1"
hmc_samples = pd.read_csv('results/' + dataset + 'RFD_hmc_samples.csv',
                          index_col=0)

fig, ax = plt.subplots(figsize=(1.65, 7 / 4))

node = 20932
p = sns.kdeplot(1 - hmc_samples.loc[node],
                linewidth=3,
                shade=True,
                alpha=1,
                kernel='gau',
                ax=ax)
ax.set_xlim([-0.001, 1.001])
ax.legend_.remove()
ax.set_xlabel(r'$p_{20932}$', labelpad=-5)
ax.set_ylabel('Prob Density', labelpad=0.1)
fig.subplots_adjust(left=0.26, bottom=0.18)
fig.savefig('plots/exampleRFD' + str(node) + 'small.pdf')

fig, ax = plt.subplots(figsize=(1.65, 7 / 4))
node = 2497
sns.kdeplot(1 - hmc_samples.iloc[node_index_inv[node]],
            linewidth=3,
            shade=True,
            alpha=1,
            kernel='gau',
            ax=ax)
ax.set_xlim([-0.005, 1.005])
ax.legend_.remove()
ax.set_xlabel(r'$p_{2497}$', labelpad=-5)
# ax.set_ylabel('Prob Density')
fig.subplots_adjust(left=0.26, bottom=0.18)
fig.savefig('plots/exampleRFD' + str(node) + 'small.pdf')

# fig.savefig('plots/exampleRFD'+str(node)+'flip.pdf')

fig, ax = plt.subplots(figsize=(1.65, 7 / 4))

node = 701
sns.kdeplot(1 - hmc_samples.iloc[node_index_inv[node]],
            linewidth=3,
            shade=True,
            alpha=1,
            kernel='gau',
            ax=ax)
ax.set_xlim([-0.005, 1.005])
ax.legend_.remove()
ax.set_xlabel(r'$p_{701}$', labelpad=-5)
# ax.set_ylabel('Prob Density')
fig.subplots_adjust(left=0.26, bottom=0.18)
fig.savefig('plots/exampleRFD' + str(node) + 'small.pdf')

# fig.savefig('plots/exampleRFD'+str(node)+'flip.pdf')
fig, ax = plt.subplots(figsize=(1.65, 7 / 4))

node = 12874
sns.kdeplot(1 - hmc_samples.iloc[node_index_inv[node]],
            linewidth=3,
            shade=True,
            alpha=1,
            kernel='gau',
            ax=ax)
ax.set_xlim([-0.005, 1.005])
ax.legend_.remove()
ax.set_xlabel(r'$p_{12874}$', labelpad=-5)
# ax.set_ylabel('Prob Density')
fig.subplots_adjust(left=0.26, bottom=0.18)
fig.savefig('plots/exampleRFD' + str(node) + 'small.pdf')
