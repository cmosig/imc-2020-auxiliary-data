import networkx as nx
from networkx.drawing import nx_agraph
from networkx.drawing.nx_agraph import to_agraph, write_dot
import config_util as confu
import configparser
from joblib import Parallel, delayed
import multiprocessing
# TODO maybe replace vp database tool with something better
import vp_database_tool as vpdb
import sys
import itertools

rfd_color = 'orangered'
maybe_rfd_color = 'gold'
no_rfd_color = 'turquoise'
peer_color = 'moccasin'
normal_as_color = 'lightskyblue'
beacon_as_color = 'plum'
selected_as_color = 'chartreuse'


def create_rfd_graph_single(peer, config, rfd_results_df_peer, all_peer_asns):
    peer_asn = vpdb.get_vp_asn(peer)

    #create one graph for each peer
    MG = nx.MultiDiGraph()  #multi and directed graph

    output_dir = config["general"]["as-graph-dir"]

    #create nodes (all ASes)
    nodes = list(
        set(itertools.chain.from_iterable(
            rfd_results_df_peer["path"].tolist())))
    beacons = eval(config["general"]["beacon-ASes"])

    for node in nodes:
        if (node == peer_asn):
            MG.add_node(node, style='filled', color=selected_as_color)
        elif (node in all_peer_asns):
            MG.add_node(node, style='filled', color=peer_color)
        elif (node in beacons):
            MG.add_node(node, style='filled', color=beacon_as_color)
        else:
            MG.add_node(node, style='filled', color=normal_as_color)

    #create edges
    #(as1,as2) -> color
    #blue for edges that have been marked both red and green
    edge_color_map = {}
    for as_path, rfd in [
            tuple(x) for x in rfd_results_df_peer[["path", "rfd"]].values
    ]:
        for i in range(len(as_path) - 1):
            pair = (as_path[i], as_path[i + 1])
            #avoid path poisining
            if (pair[0] != pair[1]):
                if (pair not in edge_color_map):
                    edge_color_map[pair] = rfd_color if rfd else no_rfd_color
                else:
                    if (edge_color_map[pair] == rfd_color):
                        edge_color_map[pair] = maybe_rfd_color if (
                            not rfd) else rfd_color
                    elif (edge_color_map[pair] == no_rfd_color):
                        edge_color_map[
                            pair] = maybe_rfd_color if rfd else no_rfd_color
    for edge, color in list(edge_color_map.items()):
        if (color != rfd_color):
            MG.add_edge(edge[1], edge[0], color=color, penwidth=2)

    for edge, color in list(edge_color_map.items()):
        if (color == rfd_color):
            MG.add_edge(edge[1], edge[0], color=color, penwidth=2)

    filename = 'as_graph_' + peer + '_AS' + peer_asn + '.pdf'
    if (maybe_rfd_color in [v for k, v in list(edge_color_map.items())]):
        print(filename)
    if (rfd_color in [v for k, v in list(edge_color_map.items())]):
        print(filename)

    A = to_agraph(MG)
    A.layout('dot')
    A.graph_attr['label'] = 'VP: ' + peer + ' (AS' + peer_asn + ')'
    A.graph_attr['labelloc'] = 't'
    A.graph_attr['fontsize'] = 20
    A.draw(output_dir + filename)
    write_dot(MG, output_dir + "as_graph_" + peer_asn + "_" + peer + ".dot")


def create_rfd_graph(configfile):

    #config variables
    config = configparser.ConfigParser()
    config.read(configfile)

    rfd_results_df = confu.get_rfd_results(config)
    peers = confu.get_list_of_relevant_peers(config)

    # # only 5min beacon
    # rfd_results_df = rfd_results_df[
    #     (rfd_results_df["prefix"] == "147.28.36.0/24")
    #     | (rfd_results_df["prefix"] == "147.28.32.0/24")
    #     | (rfd_results_df["prefix"] == "147.28.40.0/24")
    #     | (rfd_results_df["prefix"] == "147.28.44.0/24")
    #     | (rfd_results_df["prefix"] == "147.28.48.0/24")
    #     | (rfd_results_df["prefix"] == "147.28.52.0/24")
    #     | (rfd_results_df["prefix"] == "45.132.188.0/24")]

    all_peer_asns = list(map(lambda x: vpdb.get_vp_asn(x), peers))

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(create_rfd_graph_single)(
        peer=peer,
        all_peer_asns=all_peer_asns,
        config=config,
        rfd_results_df_peer=rfd_results_df[rfd_results_df["peer"] == peer])
                               for peer in peers)


create_rfd_graph(sys.argv[1])
create_all(sys.argv[1])
