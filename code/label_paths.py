from collections import Counter
import configparser
import pandas as pd
import generate_missed_received_lists as mrl
import bgpana as bap
import expected_updates as exup
import config_util as confu
import utilities as uti

uti.log("loading configs")
configfilename = 'config.ini'
config = configparser.ConfigParser()
config.read(configfilename)
start_ts = config["general"]["start-ts"]
end_ts = config["general"]["end-ts"]
prefixes = set(eval(config["general"]["prefixes"]))

# minimum time delta between end of Burst and re-advertisement
min_delta_readvertisment = int(config["general"]["min-delta-readvertisment"])

# share of Bursts that have to match the update pattern for the whole path to
# be RFD true
minimum_pattern_match = 0.9

# get timestamps for burst start based on crontab present in config
burst_starts_temp = confu.get_burst_starts(config)
burst_starts = pd.Series(burst_starts_temp, index=burst_starts_temp)
burst_length = eval(config["general"]["burst-length"])
break_length = burst_starts.iloc[1] - burst_starts.iloc[0] - burst_length


def get_burst_start(send_ts):
    # return closest burst_start from the past
    # return (send_ts - burst_starts)[lambda delta: delta > 0].idxmin()
    return min(burst_starts, key=lambda start: abs(start - send_ts + 1))


# load expected updates and get the respective burst start for each of the
# expected updates
expected_updates_list = exup.get_expected_updates('config.ini')
expected_updates_df = pd.DataFrame(expected_updates_list,
                                   columns=["prefix", "send-ts", "upd-type"])
expected_updates_df = expected_updates_df[expected_updates_df["upd-type"] ==
                                          'A']
expected_updates_df["burst-start"] = expected_updates_df["send-ts"].apply(
    get_burst_start)

# find out how many updates are sent per Burst
updates_per_burst = Counter(expected_updates_df[
    expected_updates_df["burst-start"] == burst_starts.iloc[0]]["prefix"])


def _check_update_pattern_burst(mis_rec_df):
    # this is where we actually check for the rfd signature pattern
    # input is a DataFrame of updates for a path, prefix, burst+break

    # get the prefix
    prefix = mis_rec_df["prefix"].tolist()[0]
    # get burst start
    burst_start = mis_rec_df["burst-start"].tolist()[0]
    # sanity check
    assert prefix in prefixes, "unknown prefix"
    # validate that there are no duplicates
    assert list(mis_rec_df["send-ts"]) == list(
        mis_rec_df["send-ts"].drop_duplicates()), "duplicate send-ts"

    # 1 check if re-advertisment exists
    last_sent_announcement = expected_updates_df[
        (expected_updates_df["burst-start"] == burst_start)
        & (expected_updates_df["prefix"] == prefix)]["send-ts"].max()
    if last_sent_announcement not in mis_rec_df["send-ts"].values:
        return "readv"

    # 2 time til re-advertisment is large enough
    read_delta = min(
        mis_rec_df.set_index(
            "send-ts").at[last_sent_announcement,
                          "actual_update_ts"]) - last_sent_announcement

    # readv needs to match upper bound and not be part of the next burst
    if not (min_delta_readvertisment < read_delta < break_length):
        return False
    assert (read_delta < break_length)

    # 3 At least one update missed (this is more like a sanity condition)
    if len(mis_rec_df["send-ts"]) == updates_per_burst[prefix]:
        return False

    # If everything passed return RFD True
    return True


def detect_rfd_single_new(peer):

    # load received updates for respective peer
    mis_rec_df = mrl.fast_read_mis_rec_lists(peer=peer)

    # sometimes df can be empty
    if mis_rec_df.size == 0:
        return None

    # compute missed/received ratio (this is debug info)
    # count number of received updates for each path+prefix
    mis_rec_ratios = mis_rec_df.groupby(
        ["path", "prefix"])["send-ts"].count().reset_index(name="update-count")
    # calculate the ratio between missed announcements and how many there
    # should have been
    mis_rec_ratios["mis-rec-ratio"] = mis_rec_ratios[[
        "prefix", "update-count"
    ]].apply(lambda row: ((updates_per_burst[row["prefix"]] * len(burst_starts)
                           ) - row["update-count"]) /
             (updates_per_burst[row["prefix"]] * len(burst_starts)),
             axis=1)
    mis_rec_ratios = mis_rec_ratios.drop(columns="update-count")

    # performance
    mis_rec_df.loc[:, "path"] = mis_rec_df["path"].apply(tuple)

    # find the correct burst-break phase for each send-ts
    # -> find closest burst_start which is <= beacon event
    mis_rec_df["burst-start"] = mis_rec_df["send-ts"].apply(get_burst_start)

    # detect RFD update pattern for each PATH, PREFIX (+BURST) combination
    mis_rec_df = mis_rec_df.groupby(["prefix", "path", "burst-start"])[[
        "prefix", "send-ts", "actual_update_ts", "burst-start"
    ]].apply(_check_update_pattern_burst)

    # sum results for all prefixes and paths
    mis_rec_df = mis_rec_df.groupby(
        ["prefix", "path"]).apply(tuple).reset_index(name="rfd-pattern-match")

    # RFD for prefix/path if x Bursts match the RFD pattern
    mis_rec_df["RFD"] = mis_rec_df["rfd-pattern-match"].apply(
        lambda match_list: sum([1 for match in match_list if match == True]
                               ) >= len(burst_starts) * minimum_pattern_match)

    # more debugging information
    # shows how many bursts match pattern
    mis_rec_df["match-pattern"] = mis_rec_df["rfd-pattern-match"].apply(
        lambda match_list: sum([1 for match in match_list
                                if match == True]) / len(burst_starts))

    # show how many bursts have readv missing
    mis_rec_df["readv-missing"] = mis_rec_df["rfd-pattern-match"].apply(
        lambda match_list: sum([1 for match in match_list
                                if match == "readv"]) / len(burst_starts))

    # show how many burst are empty
    mis_rec_df["empty-ratio"] = mis_rec_df["rfd-pattern-match"].apply(
        lambda match_list: len(match_list) / len(burst_starts))

    # set peer-IP
    mis_rec_df["peer-IP"] = peer

    # merge mis-rec-ratio
    mis_rec_df = mis_rec_df.merge(mis_rec_ratios, on=["path", "prefix"])

    return mis_rec_df[[
        "peer-IP", "prefix", "path", "RFD", "mis-rec-ratio", "match-pattern",
        "empty-ratio", "readv-missing"
    ]]


def detect_rfd():
    peers = confu.get_list_of_relevant_peers(config)

    # compute results
    uti.log("computing rfd_results")
    results = pd.concat(bap.paral(detect_rfd_single_new, [peers]))

    # save
    filename = 'rfd_as_path_results_' + str(start_ts) + '_' + str(end_ts)
    uti.log("saving to " + filename)
    results.to_csv(filename, sep='|', index=False, header=None)

    # save results file name to config
    config["general"]["rfd-as-path-results-file"] = filename
    with open(configfilename, 'w') as f:
        config.write(f)

    return results


detect_rfd()
