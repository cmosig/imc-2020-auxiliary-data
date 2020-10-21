import networkx as nx
from tabulate import tabulate
import pygraphviz
from networkx.drawing.nx_agraph import write_dot
import graph_investigator as g_inv
from networkx.drawing import nx_agraph
import config_util as confu
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import configparser
from joblib import Parallel, delayed
import multiprocessing
import vp_database_tool as vpdb
import sys
import itertools
import utilities as uti
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv

rfd_color = 'orangered'
maybe_rfd_color = 'gold'
no_rfd_color = 'turquoise'
peer_color = 'moccasin'
normal_as_color = 'lightskyblue'
beacon_as_color = 'plum'
selected_as_color = 'chartreuse'


def get_subgraph_of_node_neighbors(configfile, node, radius=1):
    testfile = 'test.dot'
    G = nx.MultiDiGraph(nx_agraph.read_dot(testfile))
    nodes = []
    [(nodes.append(u), nodes.append(v))
     for u, v, color in G.edges.data('color')
     if (u == node or v == node) and (color == 'green' or color == 'blue')]
    nodes = list(set(nodes))
    subgraph = G.subgraph(nodes)

    A = to_agraph(subgraph)
    A.layout('dot')
    A.draw("test_sub.svg")


def create_path_subgraph_for_node(configfile, node, rfd_results_df,
                                  complete_graph, all_paths):

    #config variables
    config = configparser.ConfigParser()
    config.read(configfile)

    all_paths_with_node = [path for path in all_paths if (node in path)]
    nodes = itertools.chain.from_iterable(all_paths_with_node)
    edges = [
        edge for edge in complete_graph.edges if
        #swapping edges because edge direction is opposite of path direction
        any([
            uti.is_subpath(full_path=path, sub_path=[edge[1], edge[0]])
            for path in all_paths_with_node
        ])
    ]

    #subgraph such that:
    #only nodes are included that are part of a path were the <node> is included
    #only edges that are part of a path were the <node> is included
    subgraph_nodes = complete_graph.subgraph(nodes)
    subgraph_edges_nodes = subgraph_nodes.edge_subgraph(edges)

    A = to_agraph(subgraph_edges_nodes)
    A.layout('dot')
    path_subgraph_root_dir = config["general"]["path-subgraph-dir"]
    A.get_node(node).attr["fillcolor"] = selected_as_color
    A.draw(path_subgraph_root_dir + "path_subgraph_" + node + ".svg")
    write_dot(subgraph_edges_nodes,
              path_subgraph_root_dir + "path_subgraph_" + node + ".dot")


def create_path_subgraph_for_all_nodes(configfile):
    #config variables
    config = configparser.ConfigParser()
    config.read(configfile)

    rfd_results_df = confu.get_rfd_results(config)
    #only 5min beacon
    rfd_results_df = rfd_results_df[
        (rfd_results_df["prefix"] == "147.28.36.0/24")
        | (rfd_results_df["prefix"] == "147.28.32.0/24")
        | (rfd_results_df["prefix"] == "147.28.40.0/24")
        | (rfd_results_df["prefix"] == "147.28.44.0/24")
        | (rfd_results_df["prefix"] == "147.28.48.0/24")
        | (rfd_results_df["prefix"] == "147.28.52.0/24")
        | (rfd_results_df["prefix"] == "45.132.188.0/24")]
    all_paths = rfd_results_df["upstream-combo"].tolist()

    nodes = list(set(itertools.chain.from_iterable(all_paths)))

    #load complete graph
    complete_graph = confu.load_complete_graph(config)

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(create_path_subgraph_for_node)(node=node,
                                               configfile=configfile,
                                               rfd_results_df=rfd_results_df,
                                               complete_graph=complete_graph,
                                               all_paths=all_paths)
        for node in tqdm(nodes))


def create_graph_for_all_peers(configfile, tooltips=False):
    #config variables
    config = configparser.ConfigParser()
    config.read(configfile)

    rfd_results_df = confu.get_rfd_results(config)
    peers = confu.get_list_of_all_peers(config)
    peer_asns = list(map(lambda x: vpdb.get_vp_asn(x), peers))
    beacons = eval(config["general"]["beacon-ASes"])

    #only 5min beacon
    rfd_results_df = rfd_results_df[
        (rfd_results_df["prefix"] == "147.28.36.0/24")
        | (rfd_results_df["prefix"] == "147.28.32.0/24")
        | (rfd_results_df["prefix"] == "147.28.40.0/24")
        | (rfd_results_df["prefix"] == "147.28.44.0/24")
        | (rfd_results_df["prefix"] == "147.28.48.0/24")
        | (rfd_results_df["prefix"] == "147.28.52.0/24")
        | (rfd_results_df["prefix"] == "45.132.188.0/24")]

    MG = nx.MultiDiGraph()  #multi and directed graph
    nodes = list(
        set(
            itertools.chain.from_iterable(
                rfd_results_df["upstream-combo"].tolist())))
    for node in nodes:
        if (node in peer_asns):
            MG.add_node(node, style='filled', color=peer_color)
        elif (node in beacons):
            MG.add_node(node, style='filled', color=beacon_as_color)
        else:
            MG.add_node(node, style='filled', color=normal_as_color)

    edge_color_map = {}
    for as_path, rfd in [
            tuple(x) for x in rfd_results_df[["upstream-combo", "rfd"]].values
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
        MG.add_edge(edge[1], edge[0], color=color)

    path_subgraph_root_dir = config["general"]["path-subgraph-dir"]
    filename = path_subgraph_root_dir + "complete_graph"
    A = to_agraph(MG)
    A.layout('dot')

    #set font
    A.graph_attr['fontname'] = "monospace"

    tooltips_nodes = {}
    if (tooltips):

        #load edge color ratios
        edge_color_ratios_file = config["general"]["edge-color-ratio"]
        unformatted_lines = open(edge_color_ratios_file,
                                 'r').read().splitlines()
        for line in unformatted_lines:
            split = line.split('|')
            tooltips_nodes[split[
                0]] = "#in: " + split[1] + '\n' + "#out: " + split[2] + '\n'
            tooltips_nodes[split[0]] += tabulate(
                [["in/out:", split[3], split[6], split[9]],
                 ["in:", split[4], split[7], split[10]],
                 ["out:", split[5], split[8], split[11]]],
                headers=["", "green", "blue", "red"])

    #link path_subgraph to nodes
    for node in nodes:
        A.get_node(node).attr["URL"] = "path_subgraph_" + node + ".svg"
        A.get_node(node).attr["fontname"] = "monospace"
        if (tooltips):
            A.get_node(node).attr["tooltip"] = tooltips_nodes[node]

    with open(filename + '.dot', 'w+') as f:
        f.write(A.to_string())
    A.draw(filename + '.svg')

    if (not tooltips):
        g_inv.as_edge_color_ratio(configfile)
        create_graph_for_all_peers(configfile, tooltips=True)


def create_cluster_graph_for_all_peers(configfile):
    #config variables
    config = configparser.ConfigParser()
    config.read(configfile)

    peers = confu.get_list_of_all_peers(config)
    rfd_results_df = confu.get_rfd_results(config)
    rfd_results_df["asn"] = rfd_results_df["upstream-combo"].apply(
        lambda x: x[0])
    peer_asns = list(map(lambda x: vpdb.get_vp_asn(x), peers))
    beacons = eval(config["general"]["beacon-ASes"])

    #only 5min beacon
    rfd_results_df = rfd_results_df[
        (rfd_results_df["prefix"] == "147.28.36.0/24")
        | (rfd_results_df["prefix"] == "147.28.32.0/24")
        | (rfd_results_df["prefix"] == "147.28.40.0/24")
        | (rfd_results_df["prefix"] == "147.28.44.0/24")
        | (rfd_results_df["prefix"] == "147.28.48.0/24")
        | (rfd_results_df["prefix"] == "147.28.52.0/24")
        | (rfd_results_df["prefix"] == "45.132.188.0/24")]

    A = pgv.AGraph(directed=True, layout='dot')
    all_asn = list(
        set(
            itertools.chain.from_iterable(
                rfd_results_df["upstream-combo"].tolist())))

    #add clusters, where cluster represents an AS
    for asn in all_asn:
        #adding all vantage points to graph including dummy
        A.add_node("r_" + asn)
        vp_in_as = []
        if (asn in peer_asns):
            vp_in_as = rfd_results_df[rfd_results_df["asn"] ==
                                      asn]["peer"].tolist()
        A.add_nodes_from(vp_in_as)
        A.add_subgraph(vp_in_as + ["r_" + asn], name="cluster_" + asn)

    #add edges
    edge_color_map = {}
    for as_path, rfd, asn, peer in [
            tuple(x) for x in rfd_results_df[
                ["upstream-combo", "rfd", "asn", "peer"]].values
    ]:
        for i in range(len(as_path) - 1):
            pair = (as_path[i], as_path[i + 1])
            #avoid path poisining
            if (pair[0] != pair[1]):
                if (i == 0 and (pair[0] in peer_asns) and (pair[0] == asn)):
                    #last cond is redundant check
                    pair = (peer, "r_" + pair[1])
                else:
                    pair = ("r_" + pair[0], "r_" + pair[1])

                if (pair not in edge_color_map):
                    edge_color_map[pair] = 'green' if rfd else 'red'
                else:
                    if (edge_color_map[pair] == 'green'):
                        edge_color_map[pair] = 'blue' if (not rfd) else 'green'
                    elif (edge_color_map[pair] == 'red'):
                        edge_color_map[pair] = 'blue' if rfd else 'red'
    for edge, color in list(edge_color_map.items()):
        A.add_edge(edge[1], edge[0], color=color)

    filename = "complete_cluster_graph"
    A.draw(filename + '.svg', prog='dot')


def create_rfd_graph_single(peer, config, rfd_results_df_peer, all_peer_asns):
    peer_asn = vpdb.get_vp_asn(peer)

    #create one graph for each peer
    MG = nx.MultiDiGraph()  #multi and directed graph

    output_dir = config["general"]["as-graph-dir"]

    #create nodes (all ASes)
    nodes = list(
        set(
            itertools.chain.from_iterable(
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
            tuple(x)
            for x in rfd_results_df_peer[["path", "rfd"]].values
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


def generate_graph_legend():
    MG = nx.MultiDiGraph()  #multi and directed graph

    MG.add_node('Beacon', style='filled', color=beacon_as_color)
    MG.add_node('AS containing VP', style='filled', color=peer_color)
    MG.add_node('Other AS1', style='filled', color=normal_as_color)
    MG.add_node('Other AS2', style='filled', color=normal_as_color)

    MG.add_edge('Beacon',
                'Other AS1',
                label='  RFD',
                color=rfd_color,
                fontsize=15,
                pensize=4)
    MG.add_edge('Other AS1',
                'Other AS2',
                label='  maybe RFD',
                color=maybe_rfd_color,
                fontsize=15,
                pensize=4)
    MG.add_edge('Other AS2',
                'AS containing VP',
                label='  no RFD',
                color=no_rfd_color,
                fontsize=15,
                pensize=4)

    A = to_agraph(MG)
    A.layout('dot')
    A.graph_attr['fontsize'] = 20
    A.draw('legend.pdf')


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
    Parallel(n_jobs=num_cores)(
        delayed(create_rfd_graph_single)(peer=peer,
                                         all_peer_asns=all_peer_asns,
                                         config=config,
                                         rfd_results_df_peer=rfd_results_df[
                                             rfd_results_df["peer"] == peer])
        #for peer in tqdm(peers))
        for peer in peers)


def create_all(configfile):
    # create_graph_for_all_peers(configfile)
    # create_path_subgraph_for_all_nodes(configfile)
    # create_cluster_graph_for_all_peers(configfile)
    create_rfd_graph(sys.argv[1])


#create_path_subgraph_for_all_nodes(configfile = 'config.ini')
#create_path_subgraph_for_node(node="3303", configfile='config.ini')
#get_subgraph_of_node_neighbors(node="2914", radius=1, configfile='config.ini')
#create_rfd_graph(sys.argv[1])
#create_graph_for_all_peers(sys.argv[1])
#create_cluster_graph_for_all_peers(sys.argv[1])

#generate_graph_legend()
create_all(sys.argv[1])
"""
def create_path_cluster_subgraph_for_node(configfile, node, rfd_results_df,
                                  complete_graph, all_paths):

    #config variables
    config = configparser.ConfigParser()
    config.read(configfile)

    all_paths_with_node = [path for path in all_paths if (node in path)]
    nodes = set(itertools.chain.from_iterable(all_paths_with_node))
    vp_ips = list(set(rfd_results_df[rfd_results_df["asn"].apply(lambda x : x in nodes)]["peer"].tolist()))
    nodes = list(map(lambda x : "r_" + x),nodes) + vp_ips
    edges = [edge for edge in complete_graph.edges if ]

    #subgraph such that:
    #only nodes are included that are part of a path were the <node> is included
    #only edges that are part of a path were the <node> is included
    subgraph_nodes = complete_graph.subgraph(nodes)
    subgraph_edges_nodes = subgraph_nodes.edge_subgraph(edges)

    A = to_agraph(subgraph_edges_nodes)
    A.layout('dot')
    path_subgraph_root_dir = config["general"]["path-subgraph-dir"]
    A.get_node(node).attr["fillcolor"] = 'yellow'
    A.draw( path_subgraph_root_dir  + "path_subgraph_" + node + ".svg" )
"""
