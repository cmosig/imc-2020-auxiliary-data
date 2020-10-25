import matplotlib
from matplotlib import rc
from matplotlib import gridspec
import matplotlib.dates as mdates
import datetime
import numpy as np
import matplotlib.pyplot as plt


def create_penalty_values(half_time,
                          update_events,
                          penalty,
                          as_timestamp=False):
    if (not as_timestamp):
        update_events = list(
            map(lambda x: int(datetime.timestamp(x)) + 7200, update_events))
    time = update_events[0]
    end = update_events[-1]
    ret = []
    timestamps = []  #some need to exist twice for plotting (penalty)
    while (time != end):
        previous_penalty = 0 if ret == [] else ret[-1]
        next_penalty = previous_penalty * np.e**(-np.log(2) / half_time)
        ret += [next_penalty]
        timestamps += [time]
        if (time in update_events):
            ret += [next_penalty + penalty]
            timestamps += [time]
        time += 1
    if (not as_timestamp):

        return list(map(lambda x: datetime.utcfromtimestamp(x),
                        timestamps)), ret
    else:
        return timestamps, ret


a_color = "mediumseagreen"
w_color = "orange"


def create_demo_plot():
    fontsize = 10
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize
    plt.tick_params(axis='both', labelsize=1)
    plt.rcParams['text.usetex'] = True

    update_interval = 420
    suppress_threshold = 3000
    reuse_limit = 750
    update_events_timestamps = [i * update_interval
                                for i in range(10)] + [4800 + 3000]
    penalty_values = create_penalty_values(
        half_time=900,
        update_events=update_events_timestamps,
        penalty=1000,
        as_timestamp=True)

    # TODO calculate these automatically. this is a pain
    t0 = 0
    t1 = 2110
    t2 = 3800
    t3 = 5800
    offset = 500
    plot_xmin = t0 - offset
    plot_xmax = t3 + offset
    plot_len = plot_xmax - plot_xmin

    # ------------------------------------------------------------
    # Both plots
    # ------------------------------------------------------------

    # constraints for both plots
    fig, ax = plt.subplots(figsize=(3.3, 2),
                           nrows=2,
                           ncols=1,
                           gridspec_kw={'height_ratios': [4, 1]})
    plt.setp(ax, xlim=(plot_xmin, plot_xmax))
    plt.subplots_adjust(hspace=0, wspace=0)

    for axes in ax:
        # display vertical lines at phases
        for t in [t0, t1, t2, t3]:
            axes.axvline(x=t, color='black', linestyle='--', linewidth='0.5')

        # show burst phase
        axes.axvspan(xmin=t0, xmax=t2, facecolor="blue", alpha=0.1)

    # ------------------------------------------------------------
    # Penalty Plot
    # ------------------------------------------------------------

    axes = ax[0]
    axes.set_ylabel("Penalty [Number]")
    axes.set_xticks([])
    threshold_lw = 1.2

    # display redline for damping phase
    axes.text(s="RFD Active", x=2950, y=200, color="black")
    axes.axhline(xmin=(t1 + offset) / plot_len,
                 y=100,
                 xmax=(t3 + offset) / plot_len,
                 color='red',
                 lw=2)

    # display line for suppress threshold
    axes.axhline(y=suppress_threshold,
                 color='r',
                 linestyle='--',
                 linewidth=threshold_lw,
                 clip_on=False,
                 xmax=1.02)
    axes.text(s="Suppress", x=t3 + 700, y=suppress_threshold - 100)

    # display line for reuse limit
    axes.axhline(y=reuse_limit,
                 color='g',
                 linestyle='--',
                 linewidth=threshold_lw,
                 clip_on=False,
                 xmax=1.02)
    axes.text(s="Reuse", x=t3 + 700, y=reuse_limit - 100)

    # penalty curve
    axes.plot(*penalty_values, linewidth=1.2, color='b')
    axes.set_ylim((0, 3600))
    axes.set_yticks([1000, 2000, 3000])

    # disable bottom edge of plot
    # axes.spines['bottom'].set_visible(False)

    # ------------------------------------------------------------
    # Update Pattern Plot
    # ------------------------------------------------------------
    marker_args = {"mew": 3, "markersize": 8}

    axes = ax[1]
    axes.set_xlabel("Time")
    axes.xaxis.set_ticks([t0, t1, t2, t3])
    axes.set_xticklabels(["$t_0$", "$t_1$", "$t_2$", "$t_3$"])
    axes.set_yticks([1, 2])
    # axes.set_yticklabels(["Updates Sent", "Updates Received"])
    axes.set_yticklabels(["Sent", "Received"])

    # Received Updates
    withdrawals = update_events_timestamps[::2]
    announcements = update_events_timestamps[1::2]
    axes.plot(announcements, [2] * len(announcements),
              '|',
              color=a_color,
              **marker_args)
    axes.plot(withdrawals, [2] * len(withdrawals),
              '|',
              color=w_color,
              **marker_args)

    # Sent Updates to peers
    withdrawals = update_events_timestamps[:6:2] + [t1 + 70]
    announcements = update_events_timestamps[1:6:2] + [t3]
    axes.plot(announcements, [1] * len(announcements),
              '|',
              color=a_color,
              **marker_args)
    axes.plot(withdrawals, [1] * len(withdrawals),
              '|',
              color=w_color,
              **marker_args)

    # disable top edge of plot
    # axes.spines['top'].set_visible(False)

    # margin to top and bottom
    axes.margins(y=0.5)

    # ------------------------------------------------------------
    # Save Plot
    # ------------------------------------------------------------

    plt.savefig("penalty_with_update_pattern.pdf", bbox_inches="tight")


create_demo_plot()
