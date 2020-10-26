import utilities as uti
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import expected_updates as exup
import pandas as pd
import generate_missed_received_lists as mrl
import configparser
import bgpana as bap
import config_util as confu
pd.options.display.float_format = '{:,.2f}'.format

# ------------------------------------------------------------
# Launch this script from one of the project directories
# (either data/march or data/april)
# ------------------------------------------------------------

fontsize = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['font.size'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.tick_params(axis='both', labelsize=1)
plt.rcParams['text.usetex'] = True

# Measurement configuration file
configfile = "config.ini"
config = configparser.ConfigParser()
config.read(configfile)

# get this
min_delta_readvertisment = int(config["general"]["min-delta-readvertisment"])
prefix_sets = eval(config["general"]["prefix-sets"])
freq_labels = eval(config["general"]["freq-labels"])

# get burst starts
burst_starts = confu.get_burst_starts(config)
burst_length = int(config["general"]["burst-length"])

# expected updates
exp_updates_glob = exup.get_expected_updates(configfile)


def _timing_stats(timings):
    return (np.mean(timings), np.std(timings))


def get_timings(series):
    assert (series["rfd"])

    damped_path = tuple(series["path"])
    vp = series["peer"]
    prefix = series["prefix"]

    exp_updates = [
        ts for (p, ts, upd) in exp_updates_glob if upd == 'A' and p == prefix
    ]

    missed_and_received_vp = pd.DataFrame(
        mrl.fast_read_mis_rec_lists(peer=vp,
                                    config=config,
                                    configfilename=configfile),
        columns=[
            "prefix", "update_ts", "update_type", "peer", "update_found",
            "path", "actual_update_ts"
        ])

    # convert path into tuple, because lists are not hashable
    missed_and_received_vp.loc[:,
                               "path"] = missed_and_received_vp["path"].apply(
                                   tuple)

    # filter by prefix and update type
    missed_and_received_vp = missed_and_received_vp[
        (missed_and_received_vp["prefix"] == prefix)
        & (missed_and_received_vp["update_type"] == 'A')]

    # get all sending timestamps for which we recived announcements
    received_updates = missed_and_received_vp[
        missed_and_received_vp["path"] == damped_path][[
            "update_ts", "actual_update_ts"
        ]]

    # sanity check
    assert (not received_updates.empty)

    time_til_damps = []
    time_til_readvs = []

    # go through each burst
    for burst_start in burst_starts:

        # get all updates within a Burst
        received_updates_burst = received_updates[
            received_updates["update_ts"].between(burst_start,
                                                  burst_start + burst_length)]

        # if no updates have been received in the Burst then simply ignore the burst
        if received_updates_burst.empty:
            continue

        # get expected updates in this burst
        exp_burst = [
            ts for ts in exp_updates
            if (ts >= burst_start) and (ts <= burst_start + burst_length)
        ]

        # DEPRECATED
        # find the first sent update for which we did not receive an
        # announcement at the vp
        # temp = set(exp_burst) - set(
        #     received_updates_burst["update_ts"].tolist())
        # time_til_damps.append(min(temp) - burst_start if len(temp) != 0 else 0)

        time_til_damps.append(
            uti.find_first_damped_update(
                exp_burst, received_updates_burst["update_ts"].tolist()) -
            burst_start)

        # get the last update and calculate the difference between the first
        # received announcement
        assert (sorted(missed_and_received_vp["update_ts"].tolist()) ==
                missed_and_received_vp["update_ts"].tolist())

        time_til_readv = min(received_updates_burst["actual_update_ts"].tolist(
        )[-1]) - received_updates_burst["update_ts"].tolist()[-1]

        # possibly not true, because don't require all bursts to match the pattern
        # assert (time_til_readv > min_delta_readvertisment)
        time_til_readvs.append(time_til_readv)

    return (_timing_stats(time_til_damps), _timing_stats(time_til_readvs))


def create_plots(path_results):
    # break_length = (burst_starts[1] - burst_starts[0]) - burst_length
    fig, ax = plt.subplots(3, 2, figsize=(3.5, 3))
    for prefix_set, ax_ in zip(prefix_sets, ax):
        ratio = 0.3

        # TIME TIL READVERTISMENT
        ax_[0].plot(
            *bap.get_cdf_space(path_results[path_results["prefix"].apply(
                lambda p: p in prefix_set)]["time_til_readv"].apply(
                    lambda x: x[0])))

        # xspace = break_length
        xspace = 90 * 60
        ax_[0].set_xlim((0, xspace))
        ax_[0].set_aspect(xspace * ratio)
        ax_[0].set_xticks(range(0, int(xspace), int(xspace / 6)))
        ax_[0].set_xticklabels(
            list(
                map(lambda s: int(s / 60),
                    range(0, int(xspace), int(xspace / 6)))))

        # TIME TIL DAMP
        ax_[1].plot(
            *bap.get_cdf_space(path_results[path_results["prefix"].apply(
                lambda p: p in prefix_set)]["time_til_damp"].apply(
                    lambda x: x[0])))

        xspace = burst_length * 0.6
        ax_[1].set_xlim((0, xspace))
        ax_[1].set_aspect(xspace * ratio)
        ax_[1].set_xticks(range(0, int(xspace), int(xspace / 6)))
        ax_[1].set_xticklabels(
            list(
                map(lambda s: int(s / 60),
                    range(0, int(xspace), int(xspace / 6)))))

    plt.setp([ax_[1] for ax_ in ax], yticklabels=[])
    plt.setp(ax[:-1], xticklabels=[])
    ax[-1][0].set_xlabel("r-delta [Minutes]")
    ax[-1][1].set_xlabel("time u. damp [Minutes]")
    # left column
    for i, ax_ in enumerate(ax):
        ax_[0].set_ylabel(
            f"{freq_labels[i].replace('Minute','').replace('s','')} [CDF]")
    # fig.autofmt_xdate()
    plt.subplots_adjust(wspace=0.05, hspace=-0.2)

    # fig.tight_layout()
    fig.savefig("dist_damp_timings.pdf", bbox_inches="tight")


def create_trimmed_plots(path_results):
    for i, prefix_set in enumerate(prefix_sets):
        fig, ax = plt.subplots(figsize=(3.3 * 0.35, 3.3 * 0.35))

        # TIME TIL READVERTISMENT
        ax.plot(*bap.get_cdf_space(path_results[path_results["prefix"].apply(
            lambda p: p in prefix_set)]["time_til_readv"].apply(
                lambda x: x[0])))

        ax.set_xlabel("r-delta [Minutes]")
        ax.set_xlim((0, 80 * 60))  # max 80 min
        ax.set_ylabel("CDF")
        x_ticks = np.array([10, 30, 60])
        ax.set_xticks(x_ticks * 60)
        ax.set_xticklabels(x_ticks)

        vline_args = {"color": "r", "linestyle": "--", "linewidth": 0.8}
        for ts in x_ticks * 60:
            ax.axvline(ts, **vline_args)

        fig.savefig(f"dist_damp_timings_readv_{i}.pdf", bbox_inches="tight")


filename = "timings_cache.pkl"
if os.path.isfile(filename):
    path_results = pd.read_pickle(filename)
else:
    path_results = confu.get_rfd_results(config)
    path_results = uti.get_paths_with_RFD(path_results)
    uti.parallel_pandas_apply(path_results, get_timings, list(path_results),
                              "timings")
    path_results[["time_til_damp", "time_til_readv"
                  ]] = pd.DataFrame(path_results["timings"].tolist(),
                                    index=path_results.index)
    del path_results["timings"]
    path_results.to_pickle(filename)

# print(path_results)
create_plots(path_results)
# create_trimmed_plots(path_results)
