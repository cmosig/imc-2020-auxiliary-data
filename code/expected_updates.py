#reads from a config.ini and return list in [(prefix,updatetime,A/W)]
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

    # crontab -> [ts]
    crontab_cache = {}

    for prefix in tqdm(prefixes):
        for update_type in ['A', 'W']:
            crontab = config[update_type + prefix]['crontab']
            if (crontab not in crontab_cache):
                crontab_cache[crontab] = uti.cron_to_ts_list(
                    crontab, start_ts, end_ts)
            for ts in crontab_cache[crontab]:
                expected_updates_list += [(prefix, ts, update_type)]

            # deprecated use crontabs
            """
            init_ts = int(config[update_type + prefix]['init-ts'])
            flap_rate = int(config[update_type + prefix]['flap-rate'])
            next_update_ts = init_ts + start_ts
            while (next_update_ts < end_ts):
                expected_updates_list.append((prefix,next_update_ts,update_type))
                next_update_ts += flap_rate
            """

    filename = 'expected_updates_' + str(start_ts) + '_' + str(end_ts)
    if (config["general"]["update-file-suffix"] != ""):
        filename += '_' + config["general"]["update-file-suffix"]

    # save expected updates in file
    open(filename, 'w+').write(str(expected_updates_list))

    # save expected updates file name to config
    config["general"]["expected-updates-file"] = filename
    with open(configfile, 'w') as f:
        config.write(f)

    return expected_updates_list


if (__name__ == "__main__"):
    # always generate new file when called from command line
    get_expected_updates(sys.argv[1], cache=False)

# print(cron_to_ts_list('10/20 0/4,1/4 * * *',1546300800,1551398400))
# cron_to_ts_list('* * * * *',1546300800,1551398400)
# pattern_list(0,59,'4/2')
# pattern_list(0,59,'*')
# pattern_list(0,23,'10/10')
# pattern_list(1,31,'1/11,2,5')
# cron_to_ts_list('10/4 * * * *')
# cron_to_ts_list('10,12,13,18 * * * *')
# print(find_highest_smaller_int(5,[12,34,56,2,55,4]))
