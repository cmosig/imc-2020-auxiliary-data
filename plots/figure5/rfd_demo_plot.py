import matplotlib
from matplotlib import rc
from matplotlib import gridspec
import matplotlib.dates as mdates
import datetime
import numpy as np
import matplotlib.pyplot as plt

a_color = "mediumseagreen"
w_color = "orange"


def burst_break_pattern_demo():
    fontsize = 10
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize
    plt.tick_params(axis='both', labelsize=1)
    plt.rcParams['text.usetex'] = True

    update_interval = 420
    bb_length = 3800 * 2
    update_events_timestamps_ = [i * update_interval for i in range(9)]

    t0_ = 0
    t1_ = 1890
    t2_ = 3580
    t3_ = 6500

    # ------------------------------------------------------------
    # All plots
    # ------------------------------------------------------------

    # constraints for both plots
    fig, axes = plt.subplots(figsize=(3.3, 0.9))
    # plt.setp(axes, xlim=(plot_xmin, plot_xmax))
    # plt.subplots_adjust(hspace=0, wspace=0)

    ticklabels = [
        "$t_0$", "$t_1$", "$t_2$", "$t_3$", "$t_4$", "$t_5$", "$t_6$", "$t_7$"
    ]
    ticks = []

    offsets = [0, bb_length]
    for offset in offsets:
        t0 = t0_ + offset
        t1 = t1_ + offset
        t2 = t2_ + offset
        t3 = t3_ + offset
        update_events_timestamps = list(
            map(lambda x: x + offset, update_events_timestamps_))
        ticks += [t0, t1, t2, t3]

        # display vertical lines at phases
        all_args = {"color": "black", "linestyle": '--', "lw": 0.3}
        for t in [t0, t2]:
            axes.axvline(x=t, **all_args, ymax=1.15, clip_on=False)
        for t in [t1, t3]:
            axes.axvline(x=t, **all_args)

        # show burst phase
        axes.axvspan(xmin=t0, xmax=t2, facecolor="blue", alpha=0.1)

        # ------------------------------------------------------------
        # Update Pattern Plot
        # ------------------------------------------------------------
        marker_args = {"mew": 1.3, "markersize": 8}

        # Beacon sending pattern

        announcements = np.array(
            update_events_timestamps) + update_interval / 2

        axes.plot(update_events_timestamps,
                  [3] * len(update_events_timestamps),
                  '|',
                  color=w_color,
                  **marker_args)

        axes.plot(announcements, [3] * (len(update_events_timestamps)),
                  '|',
                  color=a_color,
                  **marker_args)

        # no RFD
        axes.plot(announcements, [2] * len(update_events_timestamps),
                  '|',
                  color=a_color,
                  **marker_args)

        # RFD
        axes.plot(np.append(announcements[:5], t3), [1] * 6,
                  '|',
                  color=a_color,
                  **marker_args)

        # ------------------------------------------------------------
        # Annotations
        # ------------------------------------------------------------

        axes.annotate(s='',
                      xy=(t2, 1),
                      xytext=(t3, 1),
                      arrowprops={
                          "arrowstyle": '<->',
                          "lw": 0.7
                      })
        axes.text(t2 + 200, 1.25, "r-delta", fontsize=10)
        axes.text(t0 + 800, 3.55, "Burst", fontsize=10)
        axes.text(t2 + 980, 3.55, "Break", fontsize=10)

    axes.set_xlabel("Time")
    axes.xaxis.set_ticks(ticks)
    axes.set_xticklabels(ticklabels)
    axes.set_yticks([1, 2, 3])
    # axes.set_yticklabels(["RFD Path", "non-RFD Path", "Beacon Pattern"])
    axes.set_yticklabels(["RFD", "non-RFD", "Beacon"])

    # margin to top and bottom
    axes.margins(y=0.2, x=0.02)

    # ------------------------------------------------------------
    # Save Plot
    # ------------------------------------------------------------

    plt.savefig("burst_break_update_pattern.pdf", bbox_inches="tight")


burst_break_pattern_demo()
