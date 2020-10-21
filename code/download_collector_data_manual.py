import itertools
from collections import defaultdict
import shutil
import utilities as uti
import bgpana as bap
import os
import subprocess
from datetime import datetime as dt
from tqdm import tqdm


def get_url_suffixes(start_ts, end_ts):
    url_suffixes = []  #[(month,url_suffix)]

    #find first and last 5 minute mark
    first_5_min_mark = (((start_ts - 1) // 300) + 1) * 300
    last_5_min_mark = (end_ts // 300) * 300
    all_5_min_marks = [
        first_5_min_mark + 300 * i
        for i in range((last_5_min_mark - first_5_min_mark) // 300 + 1)
    ]

    #create url strings from timestamps
    #format: (yyyy_mm,yyyymmdd.HHMM.bz2)
    url_suffixes = [(dt.utcfromtimestamp(ts).strftime('%Y_%m'),
                     dt.utcfromtimestamp(ts).strftime('%Y%m%d.%H%M') + '.bz2')
                    for ts in all_5_min_marks]
    return url_suffixes


def exec_command(command):
    subprocess.Popen(command, shell=True).wait()


def download_routeviews(arguments, start_ts, end_ts):
    TAG = "Routeviews:"
    route_collectors = [
        "", "route-views.sg", "route-views.perth", "route-views.sfmix",
        "route-views.mwix", "route-views.rio", "route-views.fortaleza",
        "route-views.gixa", "route-views3", "route-views4", "route-views6",
        "route-views.amsix", "route-views.chicago", "route-views.chile",
        "route-views.eqix", "route-views.flix", "route-views.gorex",
        "route-views.isc", "route-views.kixp", "route-views.jinx",
        "route-views.linx", "route-views.napafrica", "route-views.nwax",
        "route-views.phoix", "route-views.telxatl", "route-views.wide",
        "route-views.sydney", "route-views.saopaulo", "route-views2.saopaulo"
    ]
    base_url = "http://archive.routeviews.org"
    url_suffixes = get_url_suffixes(int(start_ts), int(end_ts))

    temporary_work_directory = '.temp_download_routeviews.dump'
    if os.path.exists(temporary_work_directory):
        print("Caution! Temp dir already exists")
        return
    else:
        os.mkdir(temporary_work_directory)

    uti.log(f"{TAG} downloading files")
    commands = []
    created_files = defaultdict(list)
    for rc in route_collectors:
        for url_suffix in url_suffixes:
            # TODO make this nicer
            if rc == "":
                add = ""
            else:
                add = "/"
            url = base_url + add + rc + '/bgpdata/' + url_suffix[0].replace(
                '_', '.') + '/UPDATES/' + 'updates.' + url_suffix[1]
            temp_file_name = f"{temporary_work_directory}/{rc}_{url_suffix[1]}".replace(
                "bz2", "").strip()
            rc_arguments = ['-d', 'singlefile']
            rc_arguments.append('-o')
            rc_arguments.append('upd-file=' + url)
            bgpreader_command = 'bgpreader ' + ' '.join(
                arguments) + ' ' + ' '.join(
                    rc_arguments) + ' 2> /dev/null | gzip > ' + temp_file_name
            commands.append(bgpreader_command)
            created_files[rc].append(temp_file_name)

    bap.paral(exec_command, [tqdm(commands)])

    uti.log(f"{TAG} replacing route collector and project columns...")
    # replace column rc and rc-project by something meaningful
    for rc, filenames in created_files.items():
        for file_ in filenames:
            replace_command = f"zcat {file_}" + " | awk -F '|'  '{OFS = FS; $4=\"routeviews\"; $5=\"" + rc + "\"; print;}' " + f" | gzip > {file_}_"
            subprocess.Popen(replace_command, shell=True).wait()
            os.remove(file_)

    created_files = map(lambda fn: f"{fn}_",
                        itertools.chain.from_iterable(created_files.values()))

    uti.log(f"{TAG} merging files...")
    # merge files
    output_filename = 'routeviews_all_' + str(start_ts) + '_' + str(
        end_ts) + '.dump.gz'
    for file_ in created_files:
        merge_command = f"cat {file_} >> {output_filename}"
        subprocess.Popen(merge_command, shell=True).wait()

    uti.log(f"{TAG} merging files done")
    shutil.rmtree(temporary_work_directory)

    return output_filename


def download_isolario(arguments, start_ts, end_ts):
    TAG = "Isolario:"
    route_collectors = ["Alderaan", "Dagobah", "Korriban", "Naboo", "Taris"]
    base_url = "https://www.isolario.it/Isolario_MRT_data/"
    url_suffixes = get_url_suffixes(int(start_ts), int(end_ts))

    temporary_work_directory = '.temp_download_isolario_new.dump'
    if os.path.exists(temporary_work_directory):
        print("Caution! Temp dir already exists")
        return
    else:
        os.mkdir(temporary_work_directory)

    uti.log(f"{TAG} downloading files")
    commands = []
    created_files = defaultdict(list)
    for rc in route_collectors:
        for url_suffix in url_suffixes:
            # TODO make this nicer
            url = base_url + rc + '/' + url_suffix[
                0] + '/' + 'updates.' + url_suffix[1]
            temp_file_name = f"{temporary_work_directory}/{rc}_{url_suffix[1]}".replace(
                "bz2", "").strip()
            rc_arguments = ['-d', 'singlefile']
            rc_arguments.append('-o')
            rc_arguments.append('upd-file=' + url)
            bgpreader_command = 'bgpreader ' + ' '.join(
                arguments) + ' ' + ' '.join(
                    rc_arguments) + ' 2> /dev/null | gzip > ' + temp_file_name
            commands.append(bgpreader_command)
            created_files[rc].append(temp_file_name)

    bap.paral(exec_command, [tqdm(commands)])

    uti.log(f"{TAG} merging files...")

    uti.log(f"{TAG} replacing route collector and project columns...")
    # replace column rc and rc-project by something meaningful
    for rc, filenames in created_files.items():
        for file_ in filenames:
            replace_command = f"zcat {file_}" + " | awk -F '|'  '{OFS = FS; $4=\"Isolario\"; $5=\"" + rc + "\"; print;}' " + f" | gzip > {file_}_"
            subprocess.Popen(replace_command, shell=True).wait()
            os.remove(file_)

    created_files = map(lambda fn: f"{fn}_",
                        itertools.chain.from_iterable(created_files.values()))

    # merge files
    output_filename = 'isolario_all_' + str(start_ts) + '_' + str(
        end_ts) + '.dump.gz'
    for file_ in created_files:
        merge_command = f"cat {file_} >> {output_filename}"
        subprocess.Popen(merge_command, shell=True).wait()

    uti.log(f"{TAG} merging files done")
    shutil.rmtree(temporary_work_directory)

    return output_filename


def download_ripe_ris(arguments, start_ts, end_ts):
    TAG = "RIPE RIS:"
    route_collectors = [
        "rrc00", "rrc01", "rrc03", "rrc04", "rrc05", "rrc06", "rrc07", "rrc10",
        "rrc11", "rrc12", "rrc13", "rrc14", "rrc15", "rrc16", "rrc18", "rrc19"
        "rrc20", "rrc21", "rrc22", "rrc23", "rrc24"
    ]
    base_url = "http://data.ris.ripe.net/"
    url_suffixes = get_url_suffixes(int(start_ts), int(end_ts))

    temporary_work_directory = '.temp_download_ripe.dump'
    if os.path.exists(temporary_work_directory):
        print("Caution! Temp dir already exists")
        return
    else:
        os.mkdir(temporary_work_directory)

    uti.log(f"{TAG} downloading files")
    commands = []
    created_files = defaultdict(list)
    for rc in route_collectors:
        for url_suffix in url_suffixes:
            # TODO make this nicer
            url = base_url + rc + '/' + url_suffix[0].replace(
                '_', '.') + '/updates.' + url_suffix[1].replace('bz2', 'gz')
            temp_file_name = f"{temporary_work_directory}/{rc}_{url_suffix[1]}".replace(
                "bz2", "").strip()
            rc_arguments = ['-d', 'singlefile']
            rc_arguments.append('-o')
            rc_arguments.append('upd-file=' + url)
            bgpreader_command = 'bgpreader ' + ' '.join(
                arguments) + ' ' + ' '.join(
                    rc_arguments) + ' 2> /dev/null | gzip > ' + temp_file_name
            commands.append(bgpreader_command)
            created_files[rc].append(temp_file_name)

    bap.paral(exec_command, [tqdm(commands)])

    uti.log(f"{TAG} merging files...")

    uti.log(f"{TAG} replacing route collector and project columns...")
    # replace column rc and rc-project by something meaningful
    for rc, filenames in created_files.items():
        for file_ in filenames:
            replace_command = f"zcat {file_}" + " | awk -F '|'  '{OFS = FS; $4=\"ris\"; $5=\"" + rc + "\"; print;}' " + f" | gzip > {file_}_"
            subprocess.Popen(replace_command, shell=True).wait()
            os.remove(file_)

    created_files = map(lambda fn: f"{fn}_",
                        itertools.chain.from_iterable(created_files.values()))

    # merge files
    output_filename = 'riperis_all_' + str(start_ts) + '_' + str(
        end_ts) + '.dump.gz'
    for file_ in created_files:
        merge_command = f"cat {file_} >>  {output_filename}"
        subprocess.Popen(merge_command, shell=True).wait()

    uti.log(f"{TAG} merging files done")
    shutil.rmtree(temporary_work_directory)

    return output_filename
