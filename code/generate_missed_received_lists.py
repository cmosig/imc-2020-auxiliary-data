# generates out of the expected_updates and the actual updates a list of missed
# and received updates

import os
import bgpana as bap
import config_util as confu
from datetime import timezone
from datetime import datetime
import configparser
import expected_updates as exup
import pandas as pd
import utilities as uti

# serve expected updates from cache
expected_updates_list = exup.get_expected_updates('config.ini')
# cast into DataFrame
expected_updates_df = pd.DataFrame(expected_updates_list,
                                   columns=["prefix", "send-ts", "upd-type"])
# we are only analyzing announcements
expected_updates_df = expected_updates_df[expected_updates_df["upd-type"] ==
                                          'A']

# load config
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
output_dir = 'missed_and_received_data/'


# finds the closest expected update for the aggregator IP in given received
# update
# Aggregator IPs are set by the Beacon router and contain the point in time
# where updates are sent. Specifically the time is encoded as in:
# https://www.ripe.net/analyse/internet-measurements/routing-information-service-ris/current-ris-routing-beacons
# Sometimes the timestamp is not precise and off by up to two seconds.
# Therefore, we can't match them directly, but have to find the closest
# matching expected send timestamp.
def round_send_ts(send_ts, prefix):
    return min(exp_cache[prefix], key=lambda x: abs(x - send_ts))


def match_updates(peer, peer_subdf, output_dir, cache=True):
    uti.log(f"starting to calculate for peer {peer}")

    # filename for peer mis_rec_file
    filename = output_dir + peer + '_' + str(start_ts) + '_' + str(end_ts)

    # use existing file
    if (cache and os.path.isfile(filename)):
        # TODO
        return list(
            map(
                lambda x: (x[0], int(x[1]), x[2], x[3], eval(x[4]), eval(x[5]),
                           eval(x[6])),
                (map(lambda x: x.split('|'),
                     open(filename, 'r').read().splitlines()))))

    # only analyzing announcements
    peer_subdf = peer_subdf.copy()[peer_subdf["update"] == 'A']

    # catch if peer did not send any announcements
    if peer_subdf.size == 0:
        return None

    # clean AS-path
    peer_subdf.loc[:, "as-path"] = peer_subdf["as-path"].apply(
        lambda path: tuple(bap.clean_ASpath(path.split(' '))))

    # converting aggregator ips to actual timestamps
    # (aggregator IPs contain the send timestamps)
    peer_subdf["send-ts"] = peer_subdf[["record-ts", "aggregator-ip"]].apply(
        lambda x: uti.aggregator_ip_to_ts(
            string=x["aggregator-ip"],
            month=datetime.fromtimestamp(x["record-ts"], timezone.utc).month,
            year=datetime.fromtimestamp(x["record-ts"], timezone.utc).year),
        axis=1)

    # catch if agg-IP conversion fails
    if 0 in peer_subdf["send-ts"]:
        uti.log(
            f"There have been errors when converting agg IP for peer {peer}")

    # remove any update for which the record-ts > send-ts
    # this can happen if we receive updates from a Burst outside of the
    # current measurement interval
    peer_subdf = peer_subdf[peer_subdf["send-ts"] <= peer_subdf["record-ts"]]

    # round send-ts to nearest expected ts (see explanation above at
    # round_send_ts function)
    peer_subdf.loc[:, "send-ts"] = peer_subdf[["send-ts", "prefix"]].apply(
        lambda row: round_send_ts(row["send-ts"], row["prefix"]), axis=1)

    # group update events for PREFIX, ASPATH
    peer_subdf = peer_subdf.groupby([
        "send-ts", "as-path", "prefix"
    ])["record-ts"].apply(lambda x: sorted(list(x))).reset_index()

    # TODO remove this line maybe
    peer_subdf["update-found"] = True
    peer_subdf["peer-IP"] = peer
    peer_subdf["update"] = 'A'
    # put columns in correct order
    columns = [
        "prefix", "send-ts", "update", "peer-IP", "update-found", "as-path",
        "record-ts"
    ]
    peer_subdf = peer_subdf[columns]

    # save file. one per vantage point
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
    bap.prep_dir(output_dir)

    uti.log("computing lists now")
    # split by peer and match updates
    bap.paral(match_updates, [
        peers,
        [
            actual_updates_df[actual_updates_df["peer-IP"] == peer]
            for peer in peers
        ], [False] * len(peers)
    ])
    uti.log("computing lists done")


# filename -> file
file_cache = {}


def fast_read_mis_rec_lists(peer):
    # returns not the complete mis_received file, but only lines with rfd=True

    # generate filename from peer IP
    filename = output_dir + '/' + peer + '_' + str(start_ts) + '_' + str(
        end_ts)

    if filename in file_cache:
        return file_cache[filename]
    elif (os.path.isfile(filename)):
        # use existing file
        # TODO make nice
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


if (__name__ == "__main__"):
    generate_missed_received_list(cache=False)
