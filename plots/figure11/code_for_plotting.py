import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
# from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.tick_params(axis='both', labelsize=1)
plt.rcParams['text.usetex'] = True

df = pd.read_csv("../../data/march/rfd_paths_BeCAUSe_format_1.csv",
                 sep='|')
df['path'] = df.path.apply(eval)
df['path'] = df.path.apply(lambda x: [int(o) for o in x])
extra_params = pd.read_csv('./plotting_data_BeCAUSe.csv')
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

# init plot
fig, axes = plt.subplots(nrows=2,
                         figsize=(3.3, 4),
                         gridspec_kw={'height_ratios': [1.4, 4.5]})

ax = axes[1]
# plot ASes as scatter
colors_scatter = ["#00B0F6", "#00BF7D", "#A3A500", "#E76BF3", "#F8766D"]
g = sns.scatterplot(x="mean_hmc",
                    y='hdpi',
                    hue='category_extraflags',
                    data=extra_params,
                    palette=colors_scatter,
                    ax=ax,
                    size=7)
g.legend_.remove()

# pointers to special nodes
el = Ellipse((2, -1), 0.5, 0.5)
special_nodes = [20932, 701, 2497, 12874]
colors = ["#F8766D", "#E76BF3", "#00B0F6", "#A3A500"]
positions = [[0.27, -0.10], [-0.1, 0.05], [-0.16, 0], [0.35, 0]]

for node, color, pos in list(zip(special_nodes, colors, positions)):
    ann = ax.annotate(
        f"AS: {node}",
        xy=extra_params.iloc[node_index_inv[node]][['mean_hmc', 'hdpi']],
        xycoords='data',
        xytext=extra_params.iloc[node_index_inv[node]][['mean_hmc', 'hdpi']] -
        pos,
        size=10,
        va="center",
        bbox=dict(boxstyle="round", fc=color, ec="none"),
        arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                        fc=color,
                        ec="none",
                        patchA=None,
                        patchB=el,
                        relpos=(0.2, 0.5)))

# axis description
ax.set_xlabel('Average Probability of Damping')
ax.set_ylabel('Certainty')

# defining legend
# handles, labels = ax.get_legend_handles_labels()
# handles = handles[1:]
# labels = [
#     '1: Highly \n Unlikely', '2: Unlikely', '3: Low \n evidence', '4: Likely',
#     '5: Highly \n Likely'
# ]
# frame = True
# posi = [(0, 1.1), (0, 1), (0.3, 1), (0.62, 1.15), (0.62, 1)]
# text_pos = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
# text = ["1", "2", "3", "4", "5"]

# for label, pos, c, t, tp in zip(labels, posi, colors_scatter, text, text_pos):
#     lg = plt.legend(loc=pos,
#                     handles=[
#                         Line2D([0], [0],
#                                marker='o',
#                                markerfacecolor=c,
#                                color=c,
#                                lw=0, markersize=10)
#                     ],
#                     labels=[label],
#                     frameon=frame)
#     plt.gca().add_artist(lg)
#     ax.text(*tp, s=t)

# ax.legend(shadow=False,
#           bbox_to_anchor=(1.0, 1.22),
#           loc=1,
#           borderaxespad=0.,
#           handles=handles[1:],
#           labels=labels[1:],
#           title="Category",
#           ncol=5,
#           handletextpad=-0.4,
#           labelspacing=0.5,
#           columnspacing=0)

# treshold lines
ax.axvline(0.3, ymin=0, ymax=1, c='#cccccc')
ax.axvline(0.7, ymin=0, ymax=1, c='#cccccc')

# lims
ax.set_ylim(-0.08, 1.08)

ax = axes[0]
ax.set_yticks([])
ax.set_xticks([])
fig.subplots_adjust(hspace=0.03)

y_offset = 0.3
y_mar = 0.5 + y_offset
marker_distance = 0.2
marker_start = 0.1
marker_positions = [([marker_start + marker_distance * i], [y_mar])
                    for i in range(5)]

# ax.axhline(-0.8, c='black', lw=0.5)
# ax.axis('off')
for c, pos, i in zip(colors_scatter, marker_positions, range(1, 6)):
    ax.add_line(
        Line2D(*pos,
               marker='o',
               markerfacecolor=c,
               color=c,
               lw=0,
               markersize=10))
    ax.text(x=pos[0][0] - 0.012, y=pos[1][0] - 0.2, s=str(i))

more_args = {'ma': 'center', 'fontsize': 8}
ax.text(x=0.025, s='Highly \n Unlikely', **more_args, y=-0.85 + y_offset)
ax.text(x=0.23, s='Unlikely', **more_args, y=-0.45 + y_offset)
ax.text(x=0.42, s='Low \n Evidence', **more_args, y=-0.85 + y_offset)
ax.text(x=0.645, s='Likely', **more_args, y=-0.45 + y_offset)
ax.text(x=0.84, s='Highly \n Likely', **more_args, y=-0.85 + y_offset)
ax.text(x=0.39, y=1.3 + y_offset, s='Categories')

ax.set_ylim((-0.9, 2.4))

fig.subplots_adjust(right=0.98, top=0.85, bottom=0.12)

# save
fig.savefig(f"figure11.pdf", bbox_inches="tight")
