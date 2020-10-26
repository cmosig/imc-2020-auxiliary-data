import matplotlib.pyplot as plt
import math
import scipy
import seaborn as sns
import bgpana as bap
from collections import defaultdict
import numpy as np
import pandas as pd
import configparser

# ------------------------------------------------------------
# To get propagation timings, uncomment the below 4 lines and
# run this script from a project dir. For RIPE data
# ------------------------------------------------------------
# import generate_missed_received_lists as mrl
# configfile = "config.ini"
# config = configparser.ConfigParser()
# config.read(configfile)


def _get_propagation_time_for_peer(peer_IP):
    updates = pd.DataFrame(mrl.fast_read_mis_rec_lists(config, peer_IP,
                                                       configfile),
                           columns=[
                               "prefix", "send-ts", "upd-type", "peer-IP",
                               "found_update", "paths", "record-ts"
                           ])

    # we are only using the slow 2 hour prefixes
    slow_prefixes = {
        "45.132.191.0/24", "147.28.35.0/24", "147.28.39.0/24",
        "147.28.43.0/24", "147.28.47.0/24", "147.28.51.0/24", "147.28.55.0/24"
    }
    updates = updates[updates["prefix"].apply(
        lambda prefix: prefix in slow_prefixes)]
    # remove outlier
    # updates = updates[updates["prefix"] != "93.175.151.0/24"]

    # return if empty
    if updates.size == 0:
        return {}

    # to not use withdrawls
    updates = updates[updates["upd-type"] == 'A']

    # sanity check -> only announcements and only one IP
    assert updates["peer-IP"].nunique() <= 1, "bad missed received list"
    assert updates["upd-type"].nunique() <= 1, "bad missed received list"

    # merge record-ts for all paths
    updates = updates[["prefix", "send-ts",
                       "record-ts"]].groupby(["prefix",
                                              "send-ts"]).sum().reset_index()

    # for each Beacon event and Beacon (Prefix) we find the propagation time
    # updates = updates.drop_duplicates(["prefix", "send-ts"], keep="first")

    # calculate propation time
    updates["prop-time"] = updates["record-ts"].apply(
        lambda timestamps: sorted(timestamps)[0]) - updates["send-ts"]
    updates = updates[updates["prop-time"] >= 0]

    # calculcate average prop time for each prefix
    prop_times = updates[["prefix", "prop-time"
                          ]].groupby("prefix")["prop-time"].apply(np.mean)

    return prop_times.to_dict()


def get_and_save_average():
    all_peers = open(config["general"]["all-peers"]).read().splitlines()
    prop_times = bap.paral(_get_propagation_time_for_peer, [all_peers])

    prop_times_average = list(
        map(lambda d: str(np.median(list(d.values()))) + '\n', prop_times))
    open('average_prop_time_per_peer', 'w+').writelines(prop_times_average)


def plot_average_between_ripe_our(ripe_f, our_f):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.tick_params(axis='both', labelsize=1)
    plt.rcParams['text.usetex'] = True

    ripe_d = list(map(lambda x: float(x), open(ripe_f).read().splitlines()))
    ripe_d = [x for x in ripe_d if not math.isnan(x)]
    our_d = list(map(lambda x: float(x), open(our_f).read().splitlines()))
    our_d = [x for x in our_d if not math.isnan(x)]

    fig, ax = plt.subplots(figsize=(3.3, 3.5))
    ax.plot(*bap.get_cdf_space(ripe_d), aa=True, lw=1, label="RIPE Beacons")
    ax.plot(*bap.get_cdf_space(our_d), aa=True, lw=1, label="Anchor Prefixes")

    ax.set_xlabel("Median Propagation Time [Seconds]")
    ax.set_ylabel("CDF")
    ax.set_aspect(40)
    ax.legend(fontsize=9)
    ax.set_xlim((0, 100))
    fig.savefig("propagation_times_ripe_our_comparison.pdf",
                bbox_inches="tight")


# get_and_save_average()
plot_average_between_ripe_our("average_prop_time_per_peer_ripe_prefixes",
                              "average_prop_time_per_peer_our_beacons")
