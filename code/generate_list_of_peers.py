# generates list of peers using input-file in config

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
    command = f"zcat {input_file}" + "| awk -F '|' '{if (!seen[$9]++) {print $9}}'" + f" > {filename}"
    # save to config
    with open(configfile, 'w') as f:
        config.write(f)

    # print and execute command
    print(command)
    os.system(command)


if (__name__ == "__main__"):
    get_list_of_peers("config.ini")
