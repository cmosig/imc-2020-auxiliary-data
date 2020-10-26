import matplotlib.pyplot as plt
from scipy import stats
import random
import numpy as np
import matplotlib
from collections import Counter

# xylimits
xmin, xmax = 0, 4
ymin, ymax = 0, 100
burst_start, burst_end = xmin, xmax / 2
bin_count = 40

a_color = "mediumseagreen"


def create_plot():
    fontsize = 10
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize
    plt.tick_params(axis='both', labelsize=1)
    plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots(nrows=2, figsize=(3.3, 2))
    ax_up = ax[0]
    ax_do = ax[1]

    # ------------------------------------------------------------
    # ALL
    # ------------------------------------------------------------

    for ax_ in ax:
        # xylimits
        ax_.set_xlim(xmin, xmax)
        ax_.set_ylim(ymin, ymax)

        # show burst phase
        ax_.axvspan(xmin=burst_start,
                    xmax=burst_end,
                    facecolor="blue",
                    alpha=0.1)
        ax_.axvline(burst_end,
                    ls='--',
                    lw=1,
                    color="black",
                    clip_on=False,
                    ymax=1.13)

        # axis labels

        # ticks
        ax_.set_xticks([0, 4])
        ax_.set_xticklabels([])

    fig.subplots_adjust(hspace=0.15)

    # ------------------------------------------------------------
    # RFD PLOT (UPPER)
    # ------------------------------------------------------------

    # Burst/Break Labels
    # ax_.annotate(s='',
    #                xy=(burst_end, ymax * 0.8),
    #                xytext=(0, ymax * 0.8),
    #                arrowprops=dict(arrowstyle='<->'))
    # ax_.annotate(s='',
    #                xy=(2 * burst_end, ymax * 0.8),
    #                xytext=(burst_end, ymax * 0.8),
    #                arrowprops=dict(arrowstyle='<->'))

    height = 1.05
    shift = 0.35
    ax_up.text(burst_end * shift, ymax * height, "Burst")
    ax_up.text(burst_end + (burst_end * shift), ymax * height, "Break")
    ax_up.text(burst_end * 2, ymax / 2 - 15, "RFD", rotation=270)
    ax_do.text(burst_end * 2, ymax / 2 - 37, "non-RFD", rotation=270)

    # burst
    announcements = 75 * [0.1] + 70 * [0.2] + 55 * [0.3] + 30 * [0.4] + 28 * [
        0.5
    ] + 25 * [0.6] + 10 * [0.7] + 3 * [0.8]

    # break
    announcements += 50 * [2.5] + 28 * [2.6] + 25 * [2.7] + 10 * [2.8] + 3 * [
        2.9
    ]
    ax_up.hist(announcements,
               bins=np.linspace(xmin, xmax, num=bin_count),
               color=a_color,
               label="Announcement Count")

    # show linear regression
    x_values = np.linspace(burst_start, burst_end, bin_count // 2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x_values, [75, 70, 55, 30, 28, 25, 10, 3] + [0] * 12)
    y_values = slope * x_values + intercept
    ax_up.plot(x_values,
               y_values,
               color="blue",
               lw=2,
               label="Linear Regression of Histogram Heights")

    # ------------------------------------------------------------
    #  NON-RFD PLOT
    # ------------------------------------------------------------

    # ax_do.set_ylabel("Announcements [\#]")
    fig.text(-0.03, 0.22, s="Announcements [\#]", rotation=90)
    # burst
    announcements = random.sample(list(np.arange(0, 2, 0.0001)), 1000)
    ax_do.hist(announcements,
               bins=np.linspace(xmin, xmax, num=bin_count),
               color=a_color,
               label="Announcement Count")

    # show linear regression
    bin_heights = list(
        zip(*sorted(list(
            dict(
                Counter(
                    np.digitize(announcements,
                                np.arange(0, 2, 2 /
                                          (bin_count / 2))))).items()),
                    key=lambda x: x[0])))[1]
    x_values = np.linspace(burst_start, burst_end, bin_count // 2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x_values, bin_heights)
    y_values = slope * x_values + intercept
    ax_do.plot(x_values,
               y_values,
               color="blue",
               lw=2,
               label="Linear Regression of Histogram Heights")

    ax_do.set_xlabel("Time")
    ax_do.set_xticks([0, 0.7, 2, 2.6])
    ax_do.set_xticklabels(["$t_0$", "$t_1$", "$t_2$", "$t_3$"])

    plt.savefig("metric3_vis.pdf", bbox_inches="tight")
    plt.clf()


create_plot()
