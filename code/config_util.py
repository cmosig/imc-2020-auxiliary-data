#helps to retrieve information from the config easier

import pandas as pd
import networkx as nx
from networkx.drawing import nx_agraph
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import utilities as uti


def load_complete_graph(config):
    #load complete graph
    complete_graph_dot = config["general"][
        "path-subgraph-dir"] + "complete_graph.dot"
    return nx.MultiDiGraph(nx_agraph.read_dot(complete_graph_dot))


def get_list_of_relevant_peers(config):
    #retrieving peers to evaluate from config
    peers = []
    if (len(eval(config["general"]["peer-subset"])) > 0):
        peers = eval(config["general"]["peer-subset"])
    else:
        all_peers_file = config["general"]["all-peers"]
        peers = open(all_peers_file).read().splitlines()
    return peers


def get_list_of_all_peers(config):
    all_peers_file = config["general"]["all-peers"]
    return open(all_peers_file).read().splitlines()


def get_rfd_results(config, path=None):
    # TODO refactor function name to something with path
    #retrieve rfd results
    if path is None:
        rfd_results_file = config["general"]["rfd-as-path-results-file"]
    else:
        rfd_results_file = path + '/' + config["general"][
            "rfd-as-path-results-file"]
    # peer|prefix|upstream-combo|rfd(bool)
    rfd_results = list(
        map(lambda x: (x[0], x[1], eval(x[2]), eval(x[3]), eval(x[4])),
            (map(lambda x: x.split('|'),
                 open(rfd_results_file, 'r').read().splitlines()))))
    rfd_results_df = pd.DataFrame(
        rfd_results,
        columns=["peer", "prefix", "path", "rfd", "mis_rec_ratio"])
    return rfd_results_df


def get_indiviual_AS_results(config):
    results_file = 'rfd_individual_AS_results'
    indi_results = pd.read_csv(results_file, sep='|').fillna(0)
    return indi_results


def get_burst_starts(config):
    start_ts = int(config["general"]["start-ts"])
    end_ts = int(config["general"]["end-ts"])
    burst_starts = uti.cron_to_ts_list(
        crontab=config["general"]["burst-start"],
        start_ts=start_ts,
        end_ts=end_ts,
        progress_bar=False)
    return burst_starts


def get_rfd_node_results(path=None):
    if path is None:
        return pd.read_csv("rfd_results_nodes.csv", sep='|')
    else:
        return pd.read_csv(f"{path}/rfd_results_nodes.csv", sep='|')


def get_rfd_link_results(path=None):
    if path is None:
        df = pd.read_csv("rfd_results_links.csv", sep='|')
    else:
        df = pd.read_csv(f"{path}/rfd_results_links.csv", sep='|')
    df.loc[:, "link"] = df["link"].apply(eval)
    return df
