# reads from a config.ini and creates file with updates that are expected for
# each prefix
import configparser
from tqdm import tqdm
import utilities as uti
import sys


def get_expected_updates(configfile, cache=True):
    config = configparser.ConfigParser()
    config.read(configfile)
    start_ts = int(config["general"]["start-ts"])
    end_ts = int(config["general"]["end-ts"])
    prefixes = eval(config["general"]["prefixes"])

    # use already generated file
    expected_updates_file = config["general"]["expected-updates-file"]
    if (cache and expected_updates_file != ""):
        return eval(open(expected_updates_file).readline())

    expected_updates_list = []

    # remember the generated timestamps for each crontab
    # crontab -> [ts]
    crontab_cache = {}

    # go through each of the Beacon prefixes and analyse their crontabs
    for prefix in tqdm(prefixes):
        for update_type in ['A', 'W']:

            crontab = config[update_type + prefix]['crontab']
            if (crontab not in crontab_cache):
                crontab_cache[crontab] = uti.cron_to_ts_list(
                    crontab, start_ts, end_ts)
            for ts in crontab_cache[crontab]:
                expected_updates_list += [(prefix, ts, update_type)]

    filename = 'expected_updates_' + str(start_ts) + '_' + str(end_ts)
    if (config["general"]["update-file-suffix"] != ""):
        filename += '_' + config["general"]["update-file-suffix"]

    # save expected updates in file
    open(filename, 'w+').write(str(expected_updates_list))

    # save expected updates file name to config
    config["general"]["expected-updates-file"] = filename
    with open(configfile, 'w') as f:
        config.write(f)


if (__name__ == "__main__"):
    # always generate new file when called from command line
    get_expected_updates(sys.argv[1], cache=False)
