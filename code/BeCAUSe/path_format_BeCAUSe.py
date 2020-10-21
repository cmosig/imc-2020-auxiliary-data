import pandas as pd
import utilities as uti
import config_util as confu
import bgpana as bap
import configparser

# The output columns are supposed to be
#     RFD[True/False]|Path[ASs]|Path[Links]|Path[aggLinks]

config = configparser.ConfigParser()
config.read('config.ini')

prefix_sets = eval(config["general"]["prefix-sets"])
rfd_as_paths = confu.get_rfd_results(config)
freq_names = eval(config["general"]["freq-labels"])

for pname, prefix_set in zip(freq_names, prefix_sets):

    # only non-rfd paths were mis-rec low enough
    max_mis_rec = float(uti.get_meth_config()["m1"]["max_mis_rec_no_rfd"])
    rfd_as_paths_prefix = rfd_as_paths[(rfd_as_paths["rfd"]) | (
        rfd_as_paths["mis_rec_ratio"] <= max_mis_rec)]

    # filter paths for specific prefix set
    rfd_as_paths_prefix = rfd_as_paths_prefix[rfd_as_paths_prefix[
        "prefix"].apply(lambda prefix: prefix in prefix_set)]

    # extract links from path
    rfd_as_paths_prefix["path-with-links"] = rfd_as_paths_prefix["path"].apply(
        lambda path: tuple(bap.get_AS_links_single(path)))

    # load as relationships
    bap.init_as_rel("/home/cmosig/internet_datasets/20200301.as-rel.txt")
    rfd_as_paths_prefix["path-with-agg-links"] = rfd_as_paths_prefix[
        "path-with-links"].apply(lambda path: tuple(
            [f"{link[0]}{bap.get_relationship(link)}" for link in path]))

    # save
    rfd_as_paths_prefix.to_csv(
        f'rfd_paths_BeCAUSe_format_{pname.replace("Minute","").replace("s","").strip()}.csv',
        sep='|',
        index=False)
