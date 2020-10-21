# generates out of the expected_updates and the actual updates a list of missed
# and received updates

import os
import bgpana as bap
import config_util as confu
from datetime import timezone
from datetime import datetime
from functools import lru_cache
import configparser
import expected_updates as exup
import pandas as pd
import utilities as uti

expected_updates_list = exup.get_expected_updates('config.ini')
expected_updates_df = pd.DataFrame(expected_updates_list,
                                   columns=["prefix", "send-ts", "upd-type"])
expected_updates_df = expected_updates_df[expected_updates_df["upd-type"] ==
                                          'A']

config = configparser.ConfigParser()
config.read('config.ini')

start_ts = int(config["general"]["start-ts"])
end_ts = int(config["general"]["end-ts"])

prefixes = eval(config["general"]["prefixes"])
exp_cache = dict(
    zip(prefixes, [
        expected_updates_df[expected_updates_df["prefix"] == p]["send-ts"]
        for p in prefixes
    ]))


# @lru_cache(maxsize=30000)
def round_send_ts(send_ts, prefix):
    return min(exp_cache[prefix], key=lambda x: abs(x - send_ts))


def create_missed_and_received_single_new(peer,
                                          peer_subdf,
                                          output_dir,
                                          cache=True):
    uti.log(f"starting to calculate for peer {peer}")

    if output_dir is None:
        output_dir = 'missed_and_received_data'

    # filename for peer mis_rec_file
    filename = output_dir + '/fast/' + peer + '_' + str(start_ts) + '_' + str(
        end_ts)

    # use existing file
    if (cache and os.path.isfile(filename)):
        return list(
            map(
                lambda x: (x[0], int(x[1]), x[2], x[3], eval(x[4]), eval(x[5]),
                           eval(x[6])),
                (map(lambda x: x.split('|'),
                     open(filename, 'r').read().splitlines()))))

    # only analyzing announcements
    peer_subdf = peer_subdf.copy()[peer_subdf["update"] == 'A']

    if peer_subdf.size == 0:
        return None

    # clean AS-path
    peer_subdf.loc[:, "as-path"] = peer_subdf["as-path"].apply(
        lambda path: tuple(bap.clean_ASpath(path.split(' '))))

    # converting aggregator ips to actual timestamps
    peer_subdf["send-ts"] = peer_subdf[["record-ts", "aggregator-ip"]].apply(
        lambda x: uti.aggregator_ip_to_ts(
            string=x["aggregator-ip"],
            month=datetime.fromtimestamp(x["record-ts"], timezone.utc).month,
            year=datetime.fromtimestamp(x["record-ts"], timezone.utc).year),
        axis=1)

    if 0 in peer_subdf["send-ts"]:
        uti.log(
            f"There have been errors when converting agg IP for peer {peer}")

    # redo send-ts if record-ts > send-ts
    # DO NOT REDO, because we limit our measurements to only one month
    # peer_subdf["send-ts"] = peer_subdf[[
    #     "record-ts", "aggregator-ip", "send-ts"
    # ]].apply(lambda x: (uti.aggregator_ip_to_ts(
    #     string=x["aggregator-ip"],
    #     month=datetime.fromtimestamp(x["record-ts"], timezone.utc).month - 1,
    #     year=datetime.fromtimestamp(x["record-ts"], timezone.utc).year)) if
    #          (float(x["record-ts"]) < float(x["send-ts"])) else x["send-ts"],
    #          axis=1)

    # remove any update for which the record-ts > send-ts
    # this can happen if we receive updates from a Burst outside of the
    # current measurement interval
    peer_subdf = peer_subdf[peer_subdf["send-ts"] <= peer_subdf["record-ts"]]

    # round send-ts to nearest expected ts
    peer_subdf.loc[:, "send-ts"] = peer_subdf[["send-ts", "prefix"]].apply(
        lambda row: round_send_ts(row["send-ts"], row["prefix"]), axis=1)

    # group update events for PREFIX, ASPATH
    peer_subdf = peer_subdf.groupby([
        "send-ts", "as-path", "prefix"
    ])["record-ts"].apply(lambda x: sorted(list(x))).reset_index()

    peer_subdf["update-found"] = True
    peer_subdf["peer-IP"] = peer
    peer_subdf["update"] = 'A'
    columns = [
        "prefix", "send-ts", "update", "peer-IP", "update-found", "as-path",
        "record-ts"
    ]
    peer_subdf = peer_subdf[columns]

    peer_subdf.to_csv(filename, sep='|', header=None, index=False)


def generate_missed_received_list(cache=True):

    uti.log("initializing configs")
    peers = confu.get_list_of_relevant_peers(config)
    uti.log(f"calculating mis rec lists for {len(peers)} peers")

    actual_updates_df = pd.read_csv(config["general"]["input-file"],
                                    sep='|',
                                    header=0,
                                    names=[
                                        "update", "record-ts", "peer-IP",
                                        "prefix", "as-path", "aggregator-ip"
                                    ],
                                    usecols=[1, 2, 6, 7, 9, 14])
    actual_updates_df = actual_updates_df[(actual_updates_df["update"] == 'A')
                                          |
                                          (actual_updates_df["update"] == 'W')]
    actual_updates_df = actual_updates_df.astype({"record-ts": int})

    # a file for each peer file be stored here
    storage_missed_received = 'missed_and_received_data'
    if (not os.path.exists(storage_missed_received)):
        os.mkdir(storage_missed_received)

    uti.log("computing lists now")
    bap.paral(create_missed_and_received_single_new, [
        peers,
        [
            actual_updates_df[actual_updates_df["peer-IP"] == peer]
            for peer in peers
        ], [storage_missed_received] * len(peers), [False] * len(peers)
    ])
    uti.log("computing lists done")


# filename -> file
file_cache = {}


def fast_read_mis_rec_lists(config, peer, configfilename):
    # returns not the complete mis_received file, but only lines with rfd=True

    output_dir = 'missed_and_received_data/fast'
    filename = output_dir + '/' + peer + '_' + str(start_ts) + '_' + str(
        end_ts)

    if filename in file_cache:
        return file_cache[filename]
    elif (os.path.isfile(filename)):
        # use existing file
        ret = list(
            map(
                lambda x: (x[0], int(x[1]), x[2], x[3], eval(x[4]), eval(x[5]),
                           eval(x[6])),
                (map(lambda x: x.split('|'),
                     open(filename, 'r').read().splitlines()))))
        file_cache[filename] = ret
        return ret

    else:
        return []
        # assert False, "Missed Received File not found:" + filename


if (__name__ == "__main__"):
    generate_missed_received_list(cache=False)
