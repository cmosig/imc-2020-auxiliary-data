from datetime import datetime as dt
from joblib import Parallel, delayed
import itertools
import multiprocessing
import numpy as np
import pandas as pd
from datetime import timezone
from typing import Callable, List
from tqdm import tqdm
import configparser
import os
import sys


def remove_duplicate_lists(list_of_lists):
    list_of_lists.sort()
    return list(k for k, _ in itertools.groupby(list_of_lists))


assert (remove_duplicate_lists([[1, 2], [1, 2]]) == [[1, 2]])
assert (remove_duplicate_lists([]) == [])
assert (remove_duplicate_lists([[1, 2], [2, 3], [1, 2]]) == [[1, 2], [2, 3]])


def unix_ts_start_current_month():
    """ returns the unix ts from the beginning of the month"""
    raise DeprecationWarning
    return int((dt(dt.utcnow().year,
                   dt.utcnow().month, 1) - dt(1970, 1, 1)).total_seconds())


def unix_ts_start_month(month, year):
    """ returns the unix ts from the beginning of the supplied month"""
    return int((dt(year, month, 1) - dt(1970, 1, 1)).total_seconds())


def unix_ts_start_year(ts):
    """ return the unix ts previous beginning of year """
    year = 1970 + (int(ts) // 31536000)
    return int((dt(year, 1, 1) - dt(1970, 1, 1)).total_seconds())


def ts_to_aggregator_ip(string):
    """format specified: https://www.ripe.net/analyse/internet-measurements/routing-information-service-ris/current-ris-routing-beacons"""
    sec_since_start_of_month_bin = str(
        format(int(string) - unix_ts_start_current_month(), 'b'))
    sec_since_start_of_month_bin = fill_with_zeros(
        24, sec_since_start_of_month_bin)
    return '10.' + '.'.join(
        list(
            map(
                str,
                list(
                    map(bin_to_dec, [
                        sec_since_start_of_month_bin[i:i + 8]
                        for i in range(0, len(sec_since_start_of_month_bin), 8)
                    ])))))


def bin_to_dec(binary):
    return int(binary, 2)


def fill_with_zeros(wanted_length, binary_string):
    return '0' * (wanted_length - len(binary_string)) + binary_string


def aggregator_ip_to_ts(string,
                        month=dt.utcnow().month,
                        year=dt.utcnow().year):
    try:
        """format specified: https://www.ripe.net/analyse/internet-measurements/routing-information-service-ris/current-ris-routing-beacons"""
        string = str(string).strip()
        if (string == ''):
            return 0
        splits = string.split('.')[1:]
        if (splits == []):
            return 0
        parts = list(
            map(lambda x: fill_with_zeros(8, format(x, 'b')), map(int, splits)))
        sec_since_start_of_month = int(''.join(parts), 2)
        sec_until_start_month = unix_ts_start_month(month, year)
        return sec_since_start_of_month + sec_until_start_month
    except ValueError: 
        log(f"caught value error in aggregator_ip_to_ts: {string}")
        return 0


def cron_to_ts_list(crontab, start_ts, end_ts, progress_bar=True):
    parts = crontab.split(' ')

    if (len(parts) > 5 or len(parts) < 5):
        print('malformed crontab')

    minutes = list(map(lambda x: x * 60, pattern_list(0, 59, parts[0])))
    hours = list(map(lambda x: x * 3600, pattern_list(0, 23, parts[1])))
    days = list(
        map(lambda x: (x * 86400) - 86400, pattern_list(1, 31, parts[2])))
    months = list(map(month_num_to_seconds, pattern_list(1, 12, parts[3])))

    #weekday not implemented
    if (parts[4] != '*'):
        print('weekday not implemented.')
    #weekday = pattern_list(0,6,parts[4])

    temp_out = []
    min_year = unix_ts_start_year(start_ts)
    max_year = unix_ts_start_year(end_ts + 31536000)  #end_ts + year in s

    current_ts = 0
    prog_bar = tqdm(total=31536000, disable=not progress_bar)
    while (current_ts < 31536000):  #year in seconds
        #check if current_ts matches crontab pattern
        #print(current_ts)
        current_month = find_highest_smaller_int(current_ts, months)
        if (current_month == -1):
            current_ts += 60  #+ 1min
            prog_bar.update(60)
            continue
        temp = current_ts - current_month

        current_day = find_highest_smaller_int(temp, days)
        if (current_day == -1):
            current_ts += 60  #+ 1min
            prog_bar.update(60)
            continue
        temp -= current_day

        current_hour = find_highest_smaller_int(temp, hours)
        if (current_hour == -1):
            current_ts += 60  #+ 1min
            prog_bar.update(60)
            continue
        temp -= current_hour

        current_minute = find_highest_smaller_int(temp, minutes)
        if (current_minute == -1):
            current_ts += 60  #+ 1min
            prog_bar.update(60)
            continue
        temp -= current_minute

        if (temp == 0):
            temp_out.append(current_ts)

        current_ts += 60  #+ 1min
        prog_bar.update(60)
    prog_bar.close()

    output_list = []
    for year_num in range((max_year - min_year) // 31536000):
        output_list += list(map(lambda x: x + (year_num * 31536000), temp_out))
    output_list = list(map(lambda x: x + min_year, temp_out))
    return [i for i in output_list if (i >= start_ts and i <= end_ts)]


def pattern_list(start, end, pattern):
    if (pattern == '*'):
        return list(range(start, end + 1))
    out = []
    parts = pattern.split(',')
    for part in parts:
        if ('/' in part):
            begin, interval = part.split('/')
            cur = int(begin)
            while (cur <= end):
                out.append(cur)
                cur += int(interval)
        else:
            out.append(int(part))
    out.sort()
    return out


def month_num_to_seconds(month_num):
    """finds the number of seconds for a given month"""
    if (month_num > 12 or month_num < 1):
        print('invalid month_num: ', month_num)
    save = [0,2678400, 5097600, 7776000, 10368000, 13046400, 15638400, \
            18316800, 20995200, 23587200, 26265600, 28857600]
    return save[month_num - 1]


def find_highest_smaller_int(number, l):
    """ finds the largest number in <l> that is smaller or equal than <number> """
    candidates = [i for i in l if i <= number]
    if (candidates == []):
        return -1
    return max(candidates)


def get_burst_intervals(configfile):
    """returns burst intervals as (start,end) tuples, using datetime objects"""
    config = configparser.ConfigParser()
    config.read(configfile)
    burst_starts_crontab = config["general"]["burst-start"]
    start_ts = int(config["general"]["start-ts"])
    end_ts = int(config["general"]["end-ts"])
    burst_length = int(config["general"]["burst-length"])
    burst_starts = cron_to_ts_list(burst_starts_crontab,
                                   start_ts,
                                   end_ts,
                                   progress_bar=False)
    burst_intervals = []
    for burst_start in burst_starts:
        burst_intervals.append((dt.fromtimestamp(burst_start, timezone.utc),
                                dt.fromtimestamp(burst_start + burst_length,
                                                 timezone.utc)))
    return burst_intervals


def clean_as_path(path):
    corrected_path = []
    for i, _as in enumerate(path):
        if (len(corrected_path) > 0):
            if (_as != corrected_path[-1]):
                corrected_path.append(_as)
        else:
            corrected_path.append(_as)
    return corrected_path


def get_recursive_upstreams(as_paths, max_depth, depth=0):
    # given a set of as paths this returns all paths in the AS tree
    # all as paths need to be from one vp. eg. first element needs to be identical everywhere

    if (depth == 0):
        # clean poisoning
        corrected_paths = list(map(clean_as_path, as_paths))
    else:
        corrected_paths = as_paths

    # not checking subpaths anymore
    return remove_duplicate_lists(corrected_paths)

    # ret = []
    # for i in range(0, max_depth):
    #     corrected_paths = [x for x in corrected_paths if len(x) > i]
    #     ret += [x[:i + 1] for x in corrected_paths]

    # ret.sort()
    # ret = list(k for k, _ in itertools.groupby(ret))

    # cleaned_paths = []

    # # remove redundant paths
    # for path in ret:
    #     number_of_upper_paths = len([
    #         x for x in [
    #             is_subpath(other, path)
    #             for other in [y for y in ret if len(y) - 1 == len(path)]
    #         ] if x
    #     ])
    #     if (number_of_upper_paths != 1):
    #         cleaned_paths.append(path)

    # return cleaned_paths


assert (get_recursive_upstreams([], max_depth=1) == [])

assert (sorted(
    get_recursive_upstreams([[
        "8218",
        "8218",
        "174",
        "2914",
        "2914",
        "2914",
        "2914",
        "2497",
        "58361",
    ], [
        "174",
        "2914",
        "2914",
        "2497",
        "58361",
    ]],
                            max_depth=1)) == sorted(
                                [['8218', '174', '2914', '2497', '58361'],
                                 ['174', '2914', '2497', '58361']]))

assert (get_recursive_upstreams(
    [[
        "8218",
        "174",
        "2914",
        "2914",
        "2914",
        "2914",
        "2497",
        "58361",
    ], [
        "8218",
        "174",
        "2914",
        "2914",
        "2497",
        "58361",
    ]],
    max_depth=1) == [['8218', '174', '2914', '2497', '58361']])


def is_subpath(full_path, sub_path):
    full_path = clean_as_path(full_path)
    if (not type(full_path) == list):
        return False
    if (len(full_path) == 0 or len(sub_path) == 0):
        return False
    for i, a in enumerate(full_path):
        if (a == sub_path[0] and len(full_path) - i >= len(sub_path)):
            found = True
            for m, n in enumerate(range(i, len(sub_path) + i)):
                if (sub_path[m] != full_path[n]):
                    found = False
                    break
            if (not found):
                continue
            else:
                return True
    return False


def get_paths_with_RFD(rfd_results_df):
    return rfd_results_df[rfd_results_df["rfd"]]


def get_paths_without_RFD(rfd_results_df, max_mis_rec_no_rfd):
    rfd_no = rfd_results_df[~rfd_results_df["rfd"]]
    return rfd_no[rfd_no["mis_rec_ratio"] <= max_mis_rec_no_rfd]


def get_meth_config():
    script_dir, x = os.path.split(os.path.abspath(sys.argv[0]))
    meth_conf = configparser.ConfigParser()
    meth_conf.read(script_dir + '/methodology_parameters.ini')
    return meth_conf


def parallel_pandas_apply(df: pd.DataFrame, f: Callable, apply_to: List[str],
                          result_column: str) -> pd.DataFrame:
    """ parallelizes line wise apply on a set of columns and returns output in
    result column """
    num_cores = multiprocessing.cpu_count()
    selected_part_of_dataframe = df[apply_to]
    split_dataframe = np.array_split(selected_part_of_dataframe, num_cores * 2)

    # save in result_column
    df[result_column] = pd.concat(
        Parallel(n_jobs=num_cores)(
            delayed(lambda dfp: dfp.apply(f, axis=1))(df_part)
            for df_part in split_dataframe))
    return df


last_next_append = False


def log(message: str, next_append: bool = False):
    global last_next_append
    if next_append:
        print(str(dt.now()) + "\t| " + message, end='')
        last_next_append = True
    elif last_next_append:
        print(message)
        last_next_append = False
    else:
        print(str(dt.now()) + "\t| " + message)


def find_first_damped_update(exp_upd, act_upd):
    exp_upd.sort()
    exp_upd.reverse()
    damped_upd = act_upd[-1]
    found_before = False
    for ts in exp_upd[1:]:
        if found_before:
            break
        if ts in act_upd:
            found_before = True
        else:
            damped_upd = ts
    return damped_upd


assert find_first_damped_update([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 7]) == 5
assert find_first_damped_update([1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 7]) == 5
assert find_first_damped_update([1, 2, 3, 4, 5, 6, 7], [1, 7]) == 2
assert find_first_damped_update([1, 2, 3, 4, 5, 6, 7], [2, 7]) == 3

# test_df = pd.DataFrame({
#     'col1': random.sample(range(1, 100000), 10000),
#     'col2': random.sample(range(1, 100000), 10000)
# })
# print("test_df", test_df)
# a = parallel_pandas_apply(df=test_df,
#                           f=lambda x: x["col1"] * x["col2"],
#                           apply_to=["col1", "col2"],
#                           result_column="res")
# print(a)

# parallel_pandas_apply([1, 2, 3])
# parallel_pandas_apply(df=[4])

# print(clean_as_path(["25933", "174", "2914", "2914", "2914", "2914", "2497", "58361"]))

assert (is_subpath(
    ["25933", "174", "2914", "2914", "2914", "2914", "2497", "58361"],
    ["25933", "174", "2914", "2497", "58361"]))
assert (is_subpath(["25933", "174", "2914", "2497", "58361"],
                   ["25933", "174", "2914", "2497", "58361"]))
assert (is_subpath(["701"], ["701"]))
assert (not is_subpath([], ["701"]))
assert (not is_subpath([3], [5]))
assert (is_subpath([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5]))
assert (not is_subpath([1, 2, 3, 4, 5, 6], [3, 5]))
assert (not is_subpath([1, 2, 3, 4, 5, 6], [2, 3, 5, 6]))

# print(get_recursive_upstreams([["8218", "174", "2914", "2914", "2914", "2914", "2497", "58361",], ["8218", "174", "2914", "3130",], ["8218", "174", "701", "701", "701", "701", "2497", "58361",], ["8218", "6461", "2497", "58361",], ["8218", "6461", "2914", "2914", "2914", "2914", "2497", "2497", "58361",], ["8218", "6461", "2914", "2914", "2914", "2914", "2497", "58361",], ["8218", "6461", "2914", "3130",], ["8218", "6461", "701", "701", "701", "701", "2497", "58361"]],10))

# if path prepending of two paths, only return one
# print(get_recursive_upstreams([["13","0","3"],["13","0","3"],["13","2","3"],["13","14","24","22"],["13","14","99","68"]],10))
# print(get_recursive_upstreams([["13","14","24","22"]],10))
# print(ts_to_aggregator_ip(1567000026))
# print(unix_ts_start_year(1567412938))
# print(aggregator_ip_to_ts('10.12.26.0'))
# print(aggregator_ip_to_ts(""))
# print(aggregator_ip_to_ts(''))
# print(aggregator_ip_to_ts('10.2'))
# print(aggregator_ip_to_ts('255.255.255.255'))
# print(aggregator_ip_to_ts('93.255'))
# print(aggregator_ip_to_ts('93.1.1.1'))
