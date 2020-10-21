# downloads required bgp updates using bgpreader
# parameters based on supplied config

import subprocess
import download_collector_data_manual as down_rc
import configparser
import os
import os.path
import sys
import utilities as uti


def download_updates(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    start_ts = config["general"]["start-ts"]
    end_ts = config["general"]["end-ts"]
    prefixes = eval(config["general"]["prefixes"])

    bgpreader_arguments = []
    bgpreader_arguments.append('-w ' + str(start_ts) + ',' + str(end_ts))
    bgpreader_arguments.append('-t updates')
    for prefix in prefixes:
        bgpreader_arguments.append('-k ' + prefix)

    file_suffix = ''
    if (config["general"]["update-file-suffix"] != ""):
        file_suffix = '_' + config["general"]["update-file-suffix"]
    else:
        if (len(prefixes) > 0):
            file_suffix = ('_' + '_'.join(prefixes)).replace('/', '_')

    filename = 'all_all_updates_' + start_ts + '_' + end_ts + file_suffix + ".dump.gz"

    # setting filename in config
    config["general"]["input-file"] = filename
    with open(configfile, 'w') as f:
        config.write(f)

    # download and filter data
    routeviews_output_file = down_rc.download_routeviews(
        bgpreader_arguments, start_ts, end_ts)
    riperis_output_file = down_rc.download_ripe_ris(bgpreader_arguments,
                                                    start_ts, end_ts)
    isolario_output_file = down_rc.download_isolario(bgpreader_arguments,
                                                     start_ts, end_ts)

    # merge both
    merge_command = f"cat\
            {isolario_output_file}\
            {routeviews_output_file}\
            {riperis_output_file}\
            > {filename}"
    subprocess.Popen(merge_command, shell=True).wait()

    # remove other files
    os.remove(isolario_output_file)
    os.remove(routeviews_output_file)
    os.remove(riperis_output_file)


def query_bgpreader(arguments, start_ts, end_ts):
    output_file = 'ris-routeviews_all_updates_' + start_ts + '_' + end_ts
    bgpreader_command = 'bgpreader ' + ' '.join(arguments)
    uti.log(bgpreader_command)
    subprocess.Popen(bgpreader_command + ' > ' + output_file + ' 2> /dev/null',
                     shell=True).wait()
    return output_file


if (__name__ == "__main__"):
    download_updates(sys.argv[1])
