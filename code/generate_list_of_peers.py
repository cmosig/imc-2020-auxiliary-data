#generates list of peers using input-file in config

from tqdm import tqdm
import bgpana as bap
import subprocess
import configparser
import os
import os.path
import time
import sys


def get_list_of_peers(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    start_ts = config["general"]["start-ts"]
    end_ts = config["general"]["end-ts"]
    output_dir = config["general"]["output-dir"]
    input_file = config["general"]["input-file"]
    filename = 'all_peers_' + start_ts + '_' + end_ts
    config["general"]["all-peers"] = filename
    command = "awk -F '|' '{if (!seen[$7]++) {print $7}}'" + f" {input_file} > {filename}"
    print(command)
    os.system(command)

    good_IPs = []
    with open(filename, 'r') as f:
        all_IPs = f.read().splitlines()
        for IP in all_IPs:
            if bap.check_IPv4_syntax(IP) or bap.check_IPv6_syntax(IP):
                good_IPs.append(IP + '\n')
    open(filename, 'w').writelines(good_IPs)

    with open(configfile, 'w') as f:
        config.write(f)


if (__name__ == "__main__"):
    get_list_of_peers(sys.argv[1])
