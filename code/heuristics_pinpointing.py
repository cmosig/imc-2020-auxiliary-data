# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import utilities as uti
uti.log("Importing libaries...", next_append=True)
import statistics
from functools import lru_cache
import os
import networkx as nx
from networkx.drawing import nx_agraph
import numpy as np
import time
import expected_updates as exup
import configparser
import pandas as pd
import generate_missed_received_lists as mrl
import itertools
import multiprocessing
from collections import defaultdict
from collections import Counter
import bgpana as bap
import config_util as confu
from scipy import stats
uti.log("done ")

# ------------------------------------------------------------
# Configuration of this run
# ------------------------------------------------------------

RUN_M1_Nodes = True
RUN_M2_Nodes = True
RUN_M3_Nodes = True

RUN_M1_Links = False
RUN_M2_Links = False
RUN_M3_Links = False

NODE_SUMMARY = True
LINK_SUMMARY = False

PREFIXES_SEEN_NODES = True
PREFIXES_SEEN_LINKS = False

# ------------------------------------------------------------
# Metric Names, Filenames, Configs
# ------------------------------------------------------------

uti.log("Loading configuration files and metadata...", next_append=True)

M1, M2, M3 = "M1", "M2", "M3"

metric_names = {
    M1: "rfd_path_ratio",
    M2: "alternative_paths",
    M3: "announcement_distribution",
}

filenames_nodes = {
    M1: "m1.nodes." + metric_names[M1],
    M2: "m2.nodes." + metric_names[M2],
    M3: "m3.nodes." + metric_names[M3],
}

filenames_links = {
    M1: "m1.links." + metric_names[M1],
    M2: "m2.links." + metric_names[M2],
    M3: "m2.links." + metric_names[M3],
}

# Measurement configuration file
configfile = "config.ini"
config = configparser.ConfigParser()
config.read(configfile)

# get number of cores on machine
num_cores = multiprocessing.cpu_count()

# paramters and thresholds
meth_config = uti.get_meth_config()
max_mis_rec_no_rfd = eval(meth_config["m1"]["max_mis_rec_no_rfd"])
min_alternative_occ = eval(meth_config["m2"]["min_alternative_occ"])

# get weights for each metric to compute average between metrics
weight_metric_1 = eval(meth_config["metric-weight"]["m1"])
weight_metric_2 = eval(meth_config["metric-weight"]["m2"])
weight_metric_3 = eval(meth_config["metric-weight"]["m3"])

# get all prefix sets
temp = eval(config["general"]["prefix-sets"])
prefix_sets = list(zip(temp, list(range(len(temp)))))

# get burst starts
burst_starts = confu.get_burst_starts(config)
burst_length = int(config["general"]["burst-length"])

# get all expected updates
exp_updates_global = exup.get_expected_updates(configfile)

uti.log("done ")

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------


def get_info_update_dist(timestamps, burst_starts, burst_length):
    # total bin count. this should change if the updates/burst changes
    # TODO change that, so that it is dynamic
    bin_count = 24
    bins = np.linspace(0, burst_length, bin_count)
    bin_width = bins[1] - bins[0]
    # x values shall be in the middle of the bins
    x_values = np.asarray(list(map(lambda x: x + (bin_width / 2), bins)))

    timestamps_normalized = []
    all_y_values = []

    for burst_start in burst_starts:
        # get timestamps in burst
        timestamps_in_burst = timestamps[timestamps.between(
            burst_start,
            burst_start + burst_length)].apply(lambda x: x - burst_start)
        timestamps_normalized += timestamps_in_burst.tolist()
        timestamps_in_break = timestamps[timestamps.between(
            burst_start + burst_length,
            burst_start + 2 * burst_length)].apply(lambda x: x - burst_start)
        timestamps_normalized += timestamps_in_break.tolist()

        # can't calculate relative change if there are not updates in the
        # burst.  so go to next burst
        if (timestamps_in_burst.size <= 1):
            continue

        # each timestamp is put into a bin.
        # digitize returns bin index for each timestamp
        bin_indexes = np.digitize(timestamps_in_burst, bins)

        # count how often each index occurs
        bin_index_counter = Counter(bin_indexes)
        # [(index,index_count)]
        bin_heights = np.asarray(
            [bin_index_counter[index] for index in range(bin_count)])

        # linear regression
        slope_, intercept_, r_value, p_value, std_err = stats.linregress(
            x_values, bin_heights)
        y_values = intercept_ + slope_ * x_values
        all_y_values.append(y_values)

    # get the average linear regression line
    if all_y_values:
        y_values_merged = [
            np.median([slot_ys])
            for slot_ys in np.rot90(np.array(all_y_values))
        ]
        y_values_merged.reverse()
    else:
        y_values_merged = [0] * bin_count

    if (y_values_merged[0] != 0):
        rel_change = (y_values_merged[-1] -
                      y_values_merged[0]) / y_values_merged[0]
    else:
        rel_change = 0.0

    slope = (y_values_merged[-1] - y_values_merged[0]) / (x_values[-1] -
                                                          x_values[0])

    # this is only the projected x_hit in a really janky way
    # x_hit = y_values_merged[0] / slope
    x_hit = 0  # DEPRECATED

    return slope, x_hit, rel_change, x_values, y_values_merged, timestamps_normalized


def _get_nodes(path):
    return path


def _get_links(path):
    return list(zip(path[:-1], path[1:]))


def _get_entity_count(paths, get_entities):
    return Counter(itertools.chain.from_iterable(map(get_entities, paths)))


def _get_paths_yes_no(rfd_results):
    # get paths from rfd_results dataframe with rfd yes or no
    # also applying minimum visibilty for rfd no paths
    paths_yes = [
        k for k, _ in itertools.groupby(
            uti.get_paths_with_RFD(rfd_results)["path"].tolist())
    ]
    paths_no = [
        k for k, _ in itertools.groupby(
            uti.get_paths_without_RFD(rfd_results, max_mis_rec_no_rfd)
            ["path"].tolist())
    ]
    return paths_yes, paths_no


# @profile
def _get_alternative_paths_for_single_rfd_path(series):
    # TODO you could think about throwing away paths with two few updates
    # overall
    if not series["rfd"]:
        return []

    damped_path = tuple(series["path"])
    vp = series["peer"]
    prefix = series["prefix"]

    exp_updates = [
        ts for (p, ts, upd) in exp_updates_global if upd == 'A' and p == prefix
    ]

    # retrieve received updates for vantage point
    missed_and_received_vp = mrl.fast_read_mis_rec_lists(vp)

    # convert path into tuple, because lists are not hashable
    missed_and_received_vp.loc[:,
                               "path"] = missed_and_received_vp["path"].apply(
                                   tuple)

    # filter by prefix and update type
    missed_and_received_vp = missed_and_received_vp[
        (missed_and_received_vp["prefix"] == prefix)
        & (missed_and_received_vp["update_type"] == 'A')]

    # get all sending timestamps for which we recived announcements
    received_updates = missed_and_received_vp[missed_and_received_vp["path"] ==
                                              damped_path]["send-ts"]

    # filter by prefix and update type
    alternative_path_candidates = missed_and_received_vp[
        missed_and_received_vp["path"] != damped_path]

    # sanity check
    assert (not received_updates.empty)

    # ratios for each alternative path is stored here. default is []
    altpath_ratios = defaultdict(list)

    # go through each burst
    for burst_start in burst_starts:

        # get all updates within a Burst
        received_updates_burst = received_updates[received_updates.between(
            burst_start, burst_start + burst_length)].tolist()

        # get expected updates in this burst
        exp_burst = [
            ts for ts in exp_updates
            if (ts >= burst_start) and (ts <= burst_start + burst_length)
        ]

        # find the first sent update for which we did not receive an
        # announcement at the vp
        # temp = set(exp_burst) - set(received_updates_burst)
        # time_til_damp = min(temp) - burst_start if len(temp) != 0 else 0

        if len(received_updates_burst) > 0:
            # it can be that for some bursts no update are received
            time_til_damp = uti.find_first_damped_update(
                exp_burst, received_updates_burst) - burst_start
        else:
            # uti.log(
            #     f"setting ttd to 0, for burst_start {burst_start} for {series}"
            # )
            time_til_damp = 0

        # if we have never received updates within a Burst for the damped
        # path then take all paths
        # if (len(received_updates_burst) > 0):
        #     for expected_update in exp_burst:
        #         if expected_update not in received_updates_burst:
        #             time_til_damp = expected_update - burst_start
        #             break

        # take only updates that are in the Burst
        alternative_path_candidates_burst = alternative_path_candidates[
            alternative_path_candidates["send-ts"].between(
                burst_start, burst_start + burst_length)]

        # if there were no alternative path candidates in this burst then go to
        # the next burst
        if (alternative_path_candidates_burst.empty):
            continue

        # count for each path how many announcements were received overall and
        # after damp
        # NOTE: count() does not count NaNs, size() does
        update_counts_burst = alternative_path_candidates_burst[[
            "path", "send-ts"
        ]].groupby(["path"]).count()

        update_counts_after_damp = alternative_path_candidates_burst[
            alternative_path_candidates_burst["send-ts"] > burst_start +
            time_til_damp][["path", "send-ts"]].groupby(["path"]).count()

        # if there are no updates after the damp then go to next burst
        if (update_counts_after_damp.empty):
            continue

        # create merged df
        merged_update_counts = update_counts_after_damp.join(
            update_counts_burst,
            how="inner",  # -> intersection of keys
            lsuffix="_after",
            rsuffix="_all")

        # normalize by leftover time betwen damp and end of burst
        merged_update_counts["send-ts_after_norm"] = merged_update_counts[
            "send-ts_after"] / (burst_length - time_til_damp)

        # normalize by length of burst
        merged_update_counts["send-ts_all_norm"] = merged_update_counts[
            "send-ts_all"] / burst_length

        # calculate ratio between updates/s after damp and overall
        #
        # ratio = 1 : updates/s are equal before damp and overall
        # ratio < 1 : fewer updates/s for that path after damp
        # ratio > 1 : more updates for that path after damp
        merged_update_counts["ratio"] = merged_update_counts[
            "send-ts_after_norm"] / merged_update_counts["send-ts_all_norm"]

        # BAD! YOU FOOL MARCIN IS COOL!
        # altpath_ratios_burst = merged_update_counts[[
        #     "ratio"
        # ]].T.to_dict("records")[0].items()

        altpath_ratios_burst = merged_update_counts["ratio"].to_dict()

        # save ratio for alternative path
        for path, ratios in altpath_ratios_burst.items():
            assert (ratios != [])
            altpath_ratios[path].append(ratios)

    # get all path with average ratio higher than specified min_alternative_occ
    # (=1). Or in other wortds get all paths for which we received equally as
    # many updates after the time of damping as overall
    # alternative_paths = [
    #     path for path, ratios in altpath_ratios.items()
    #     if sum(ratios) / len(burst_starts) > min_alternative_occ
    # ]
    alternative_paths = [
        path for path, ratios in altpath_ratios.items()
        if sum(ratios) / len(burst_starts) > min_alternative_occ
    ]
    return alternative_paths


def _fill_with_alternative_paths(rfd_results):
    # TEMP for Debugging:
    # rfd_results[rfd_results["rfd"]].head(50).apply(
    #     _get_alternative_paths_for_single_rfd_path, axis=1)

    rfd_results = uti.parallel_pandas_apply(
        rfd_results, _get_alternative_paths_for_single_rfd_path,
        ["path", "peer", "prefix", "rfd"], "alternative-paths")

    # print(rfd_results[rfd_results.rfd][["path",
    #                                     "alternative-paths"]])
    return rfd_results


@lru_cache(maxsize=1000)
def _get_subgraph(vp):
    directory = "as_graphs/"
    files = pd.Series(os.listdir(directory))
    files = files[files.apply(
        lambda filename: vp in filename and 'dot' in filename)]
    subgraphs = []
    for file_ in files:
        subgraphs.append(nx.MultiDiGraph(nx_agraph.read_dot(directory +
                                                            file_)))
    return nx.compose_all(subgraphs)


def _create_alternative_path_dict(rfd_results):

    # select only rfd paths
    temp = rfd_results[rfd_results.rfd].copy()

    # make list a tuple for hashing
    temp.loc[:, "alternative-paths"] = temp["alternative-paths"].apply(tuple)

    # merge alternative paths if there are multiple identical RFD paths caused
    # by multiple VPs in a single VP AS
    temp = temp.groupby("path")["alternative-paths"].apply(lambda x: list(
        set(itertools.chain.from_iterable(x)))).to_dict().items()

    return temp


@lru_cache(maxsize=50000)
def _clean_path_and_get_entities(path, get_entities):
    return get_entities(bap.clean_ASpath(path.split(' ')))


# ------------------------------------------------------------
# Core Functions to Calculate Metrics
# ------------------------------------------------------------


def _M1_rfd_path_ratio(get_entities, paths_yes, paths_no):
    # input:
    # - get_entities: a function that gets all relavant entities from a path.
    # may it be nodes or edges
    # - paths_yes/no: paths labeled with property yes or no
    #
    # output:
    # - dictionary of all present entities with their path ratio

    # getting how often each entity occuring in yes/no paths
    yes_count = defaultdict(int, _get_entity_count(paths_yes, get_entities))
    no_count = defaultdict(int, _get_entity_count(paths_no, get_entities))

    # calculate metric1 for all entities that occured in either type of paths
    return dict(
        map(
            lambda ent: (ent, (yes_count[ent] /
                               (yes_count[ent] + no_count[ent]))),
            set(yes_count.keys()) | set(no_count.keys())))


def _M2_alternative_paths(get_entities, rfd_path_alternative_paths):
    # [(damped_path, [alternative paths])]
    # input: RFD paths
    # output: alternative path score for each entity in a defaultdict
    # (default = 0)

    results_for_each_path = {}

    for damped_path, altpaths in rfd_path_alternative_paths:
        # sanity check. beacon and vp should be the same
        assert (all([
            damped_path[0] == alt[0] and damped_path[-1] == alt[-1]
            for alt in altpaths
        ]))

        if len(altpaths) > 0:
            # If there are alternative paths, derive damper from that.
            candidates = []
            for altpath in altpaths:
                candidates += list(
                    set(get_entities(damped_path)) -
                    set(get_entities(altpath)))

            # I return how often each entity has been classified as candidate
            # normalized by the number of alternative paths.
            normalized_candidate_counts = dict(
                map(
                    lambda ent_count:
                    (ent_count[0], ent_count[1] / len(altpaths)),
                    Counter(candidates).items()))

            # checking that normalized counts are between 1 and 0
            assert (all([
                0 < norm_candidate_count <= 1 for norm_candidate_count in
                normalized_candidate_counts.values()
            ]))

            # there should only be one rfd-path alternative path pair
            assert (damped_path not in results_for_each_path)
            results_for_each_path[damped_path] = normalized_candidate_counts
        else:
            vp_AS = damped_path[0]
            beacon_AS = damped_path[-1]

            # 1. build up directed multigraph between vp and beacon
            # merged graphs with the same vp asn
            graph = _get_subgraph(vp_AS)
            # path exist
            assert (nx.shortest_path(graph, beacon_AS, vp_AS))

            # if there are no alternative paths, the vp could always be the
            # damper, i.e., first entity
            tagged_ents = []

            for ent in get_entities(damped_path):
                # 2.1. go through each AS on the damped path and remove from
                # graph
                temp_graph = graph.copy()

                # remove link or AS:
                if isinstance(ent, tuple):
                    temp_graph.remove_edge(ent[1], ent[0])
                else:
                    temp_graph.remove_node(ent)

                # when doing this for ASs we remove the VP as well. To
                # save us from an execption later we catch the case here
                if not temp_graph.has_node(vp_AS):
                    tagged_ents.append(vp_AS)
                    continue

                # Beacon cannot be the damper. Therefore if removed from graph,
                # then just continue
                if not temp_graph.has_node(beacon_AS):
                    continue

                # 2.2. check if there is a path from becon to vp without that
                # one entity (Link/AS)
                try:
                    nx.shortest_path(temp_graph, beacon_AS, vp_AS)
                except nx.NetworkXNoPath:
                    # -> if no then save that AS
                    tagged_ents.append(ent)
                    continue

                assert ent not in tagged_ents, "duplicate entities on path"

            # distribute a score of one
            score_for_each_ent = 1

            # sanity checks
            assert beacon_AS not in tagged_ents, "ya Beacon is damping. go home"
            assert len(tagged_ents) <= len(
                damped_path) - 1, "you found more damper than are on the path"
            assert damped_path not in results_for_each_path, "duplicate paths"

            # save scores for each AS on damped path
            results_for_each_path[damped_path] = dict(
                zip(tagged_ents, [score_for_each_ent] * len(tagged_ents)))

    # get the sum of all scores for each an normalize these by the number of
    # rfd paths the respective entity has been on
    # [(ent, score)]
    merged_scores = list(
        itertools.chain.from_iterable(
            map(lambda res_dict: res_dict.items(),
                results_for_each_path.values())))

    ent_scores = defaultdict(lambda: 0)

    # sum score for each AS
    for ent, score in merged_scores:
        ent_scores[ent] += score

    # calculate how often each entity occured on rfd paths
    all_rfd_paths = tuple(zip(*rfd_path_alternative_paths))[0]
    path_counter = Counter(
        itertools.chain.from_iterable(map(get_entities, all_rfd_paths)))

    # normalize each score by the number of as paths the ent is present on
    for ent, score in ent_scores.items():
        ent_scores[ent] = score / path_counter[ent]

    # check that all score are between 0 and 1
    assert (all([0 <= score <= 1 for score in ent_scores.values()]))

    return ent_scores


def _M3_announcement_distribution(get_entities, raw_updates):
    # raw_updates: "type", "timestamp", "prefix", "path"
    # calculates the announcement distribution score for each entitiy

    # no pain much gain
    raw_updates = raw_updates.dropna().copy()

    raw_updates["entities"] = raw_updates["path"].apply(
        lambda path: tuple(_clean_path_and_get_entities(path, get_entities)))

    #   1: get all entities
    all_entities = set.union(
        *map(set, raw_updates["entities"].drop_duplicates()))

    #   2: calculate distribution for each entitiy
    # get timestamps for each ent
    ent_timestamps = raw_updates.explode("entities")[[
        "timestamp", "entities"
    ]].groupby("entities")["timestamp"].apply(tuple).to_dict()
    assert all(
        len(timestamps) > 0
        for timestamps in ent_timestamps.values()), "timestamps got lost"

    disribution_results = bap.paral(get_info_update_dist, [
        list(map(pd.Series, ent_timestamps.values())), [burst_starts] *
        len(ent_timestamps), [burst_length] * len(ent_timestamps)
    ])

    # in distribution results:
    # slope, x_hit, rel_change, x_values, y_values_merged, timestamps_normalized
    # we only really care about rel_change and slope
    disribution_results_shrunk = dict(
        zip(ent_timestamps.keys(),
            [(x[0], x[2]) for x in disribution_results]))

    #   3: calculate relative change upper bound
    #   calculating the 95th percentile of all negative values negated
    all_negative_rel_change_values = [
        rel_change for _, rel_change in disribution_results_shrunk.values()
        if rel_change < 0
    ]
    assert len(all_negative_rel_change_values) <= len(
        disribution_results_shrunk), "lengths don't make sense"
    rel_change_upper_bound = np.percentile(
        list(map(abs, all_negative_rel_change_values)), 95)
    assert rel_change_upper_bound > 0, "relative change upper bound is greater than zero"
    assert rel_change_upper_bound < abs(
        min(all_negative_rel_change_values)
    ), "relative change upper bound is greater than negated smalled relative change"

    #   4: compute score based on relative change
    scores = {}

    for entity, (slope, relative_change) in disribution_results_shrunk.items():
        # if relative change is > 0 the probabilty should be zero
        # if relative change is <= 0 apply function taking 95 percentile of

        # this should not happen. see notes
        assert not (relative_change > 0
                    and slope < 0), "relative change and score are strange"

        if (relative_change > 0 or slope > 0):
            score = 0.0
        elif (abs(relative_change) >= rel_change_upper_bound):
            score = 1.0
        else:
            # normalize by upper bound
            score = abs(relative_change) / rel_change_upper_bound

        assert score >= 0, "score is smaller than zero"
        assert score <= 1, "score is greater than one"
        assert entity not in scores, "duplicate entity"

        scores[entity] = score

    assert len(scores) == len(disribution_results), "entities got lost"
    assert len(scores) == len(all_entities), "entities got lost"

    return scores


def _pref_seen(get_entities, rfd_results):
    # get all relevant paths
    res = pd.concat([
        uti.get_paths_with_RFD(rfd_results),
        uti.get_paths_without_RFD(rfd_results, max_mis_rec_no_rfd)
    ])
    res.loc[:, "path"] = res["path"].apply(get_entities)
    res = res[["path",
               "prefix"]].explode("path").groupby("path")["prefix"].nunique()

    return res.to_dict()


# ------------------------------------------------------------
# Wrapper Functions to Feed Core Functions
# ------------------------------------------------------------


def _links_M1_rfd_path_ratio(rfd_results):
    # prepare data to run in metric 1 function
    paths_yes, paths_no = _get_paths_yes_no(rfd_results)
    return _M1_rfd_path_ratio(_get_links, paths_yes, paths_no)


def _links_M2_alternative_paths(rfd_results):
    return _M2_alternative_paths(_get_links,
                                 _create_alternative_path_dict(rfd_results))


def _links_M3_announcement_distribution(raw_updates):
    return _M3_announcement_distribution(_get_links, raw_updates)


def _nodes_M1_rfd_path_ratio(rfd_results):
    # prepare data to run in metric 1 function
    paths_yes, paths_no = _get_paths_yes_no(rfd_results)
    return _M1_rfd_path_ratio(_get_nodes, paths_yes, paths_no)


def _nodes_M2_alternative_paths(rfd_results):

    # call metric 2 with a list of the damped path with their alternative paths
    return _M2_alternative_paths(_get_nodes,
                                 _create_alternative_path_dict(rfd_results))


def _nodes_M3_announcement_distribution(raw_updates):
    return _M3_announcement_distribution(_get_nodes, raw_updates)


def _nodes_pref_seen(rfd_results):
    return _pref_seen(_get_nodes, rfd_results)


def _links_pref_seen(rfd_results):
    return _pref_seen(_get_links, rfd_results)


# ------------------------------------------------------------
# Sub-Main Functions
# ------------------------------------------------------------


def link_analysis(rfd_results, raw_updates):
    if rfd_results is None:
        rfd_results = confu.get_rfd_results(config)
        rfd_results.loc[:, "path"] = rfd_results["path"].apply(tuple)
        assert (not RUN_M2_Links)

    results = {}
    for prefix_set, set_id in prefix_sets:
        uti.log(f"Running for prefix set: {prefix_set}")
        M1_results, M2_results, M3_results = None, None, None
        # filter RFD results to only contain prefixes in the prefix set
        prefix_set_rfd_results = rfd_results[rfd_results["prefix"].apply(
            lambda prefix: prefix in prefix_set)]

        if RUN_M3_Links:
            prefix_set_raw_updates = raw_updates[raw_updates["prefix"].apply(
                lambda prefix: prefix in prefix_set)]

        # all these functions return a dict with link -> score

        if RUN_M1_Links:
            uti.log("Links: M1: RFD path ratio...", next_append=True)
            M1_results = _links_M1_rfd_path_ratio(prefix_set_rfd_results)
            uti.log("done")
        if RUN_M2_Links:
            uti.log("Links: M2: alternative path evaluation...",
                    next_append=True)
            M2_results = _links_M2_alternative_paths(prefix_set_rfd_results)
            uti.log("done")
        if RUN_M3_Links:
            uti.log("Links: M3: announcement distribution...",
                    next_append=True)
            M3_results = _links_M3_announcement_distribution(
                prefix_set_raw_updates)
            uti.log("done")
        if PREFIXES_SEEN_LINKS:
            uti.log("Links: Prefixes seen")
            pref_seen = _links_pref_seen(prefix_set_rfd_results)

        results[set_id] = (M1_results, M2_results, M3_results, pref_seen)
    return results


def node_analysis(rfd_results, raw_updates):
    if rfd_results is None:
        rfd_results = confu.get_rfd_results(config)
        rfd_results.loc[:, "path"] = rfd_results["path"].apply(tuple)
        assert (not RUN_M2_Nodes)

    results = {}
    for prefix_set, set_id in prefix_sets:
        uti.log(f"Running for prefix set: {prefix_set}")
        M1_results, M2_results, M3_results = None, None, None
        # filter RFD results to only contain prefixes in the prefix set
        prefix_set_rfd_results = rfd_results[rfd_results["prefix"].apply(
            lambda prefix: prefix in prefix_set)]

        if RUN_M3_Nodes:
            prefix_set_raw_updates = raw_updates[raw_updates["prefix"].apply(
                lambda prefix: prefix in prefix_set)]

        # all these functions return a dict with node -> score

        if RUN_M1_Nodes:
            uti.log("Nodes: M1: RFD path ratio...", next_append=True)
            M1_results = _nodes_M1_rfd_path_ratio(prefix_set_rfd_results)
            uti.log("done")

        if RUN_M2_Nodes:
            uti.log("Nodes: M2: alternative path evaluation...",
                    next_append=True)
            M2_results = _nodes_M2_alternative_paths(prefix_set_rfd_results)
            uti.log("done")

        if RUN_M3_Nodes:
            uti.log("Nodes: M3: announcement distribution...",
                    next_append=True)
            M3_results = _nodes_M3_announcement_distribution(
                prefix_set_raw_updates)
            uti.log("done")

        if PREFIXES_SEEN_NODES:
            uti.log("Nodes: Prefixes seen")
            pref_seen = _nodes_pref_seen(prefix_set_rfd_results)

        results[set_id] = (M1_results, M2_results, M3_results, pref_seen)
    return results


def _generic_summary(results):
    # (pref_id -> M* -> entitiy_results)

    prefix_dfs = []

    for pset, pref_id in prefix_sets:
        M1, M2, M3, pref_counts = results[pref_id]

        assert M1 is not None or M2 is not None or M3 is not None\
                or pref_counts is not None,\
                "you did not calculate all metrics you fool"

        M1_df = pd.DataFrame.from_dict(M1, orient="index", columns=["M1"])
        M2_df = pd.DataFrame.from_dict(M2, orient="index", columns=["M2"])
        M3_df = pd.DataFrame.from_dict(M3, orient="index", columns=["M3"])
        pref_counts_df = pd.DataFrame.from_dict(pref_counts,
                                                orient="index",
                                                columns=["pcount"])

        # similar thing, but multiindex from tuples
        # to_compare = pd.concat([M1_df, M2_df, M3_df], axis=1,
        # join="outer").sort_index()

        # join Metric dataframes
        pref_set_df = M1_df.join(M2_df, how="outer").join(
            M3_df, how="outer").join(pref_counts_df, how="outer")

        # fill NaNs with 0, because this means they either not been on any RFD
        # paths or have just not been detected. Either means 0 is the logical
        # value
        pref_set_df = pref_set_df.fillna(0)

        # compute the average
        pref_set_df["metric_average"] = pref_set_df[["M1", "M2",
                                                     "M3"]].mean(axis=1)

        # create pref_set_ID column and fill with current ID
        pref_set_df["pID"] = pref_id

        # store
        prefix_dfs.append(pref_set_df)

    return pd.concat(prefix_dfs)


def node_summary(node_results):
    res = _generic_summary(node_results)
    res.index.name = 'node'
    res.to_csv('rfd_results_nodes.csv', sep='|')


def link_summary(link_results):
    res = _generic_summary(link_results)
    res.index.name = 'link'
    res.to_csv('rfd_results_links.csv', sep='|')


# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------


def main():
    rfd_results = None
    raw_updates = None

    if RUN_M3_Nodes or RUN_M3_Links:
        raw_updates = pd.read_csv(
            config["general"]["input-file"],
            sep='|',
            header=None,
            usecols=[1, 2, 9, 11],
            names=["type", "timestamp", "prefix", "path"])

    if (RUN_M2_Links or RUN_M2_Nodes):
        uti.log("Finding alternative paths for damped paths...",
                next_append=True)
        rfd_results = confu.get_rfd_results(config)
        rfd_results.loc[:, "path"] = rfd_results["path"].apply(tuple)
        rfd_results = _fill_with_alternative_paths(rfd_results)
        uti.log("done")

    node_results = None
    if (RUN_M1_Nodes or RUN_M2_Nodes or RUN_M3_Nodes or PREFIXES_SEEN_NODES):
        uti.log("Running NODE analysis...")
        node_results = node_analysis(rfd_results, raw_updates)
        uti.log("NODE analysis done")

    link_results = None
    if (RUN_M1_Links or RUN_M2_Links or RUN_M3_Links or PREFIXES_SEEN_LINKS):
        uti.log("Running LINK analysis")
        link_results = link_analysis(rfd_results, raw_updates)
        uti.log("LINK analysis done")

    if (NODE_SUMMARY):
        uti.log("Computing NODE Summary")
        node_summary(node_results)
        uti.log("NODE Summary done")

    if (LINK_SUMMARY):
        uti.log("Computing LINK Summary")
        link_summary(link_results)
        uti.log("LINK Summary done")

main()
