from typing import Tuple, Iterable, Set, List, Callable
from tqdm import tqdm
from datetime import datetime as dt
import time
from collections import defaultdict
import os
import numpy as np
# import validators
from joblib import Parallel, delayed
from os import path
import itertools

ASN = int
ASpath = List[ASN]
ASlink = Tuple[ASN, ASN]

as_rel_dict = None
as_rank_dict = None
as_country_codes_dict = None


def get_AS_links(aspaths: Iterable[ASpath]) -> Set[ASlink]:
    return set(
        itertools.chain.from_iterable(
            [list(zip(aspath[:-1], aspath[1:])) for aspath in aspaths]))


def get_AS_links_single(aspath: ASpath) -> List[ASlink]:
    return list(zip(aspath[:-1], aspath[1:]))


def rsp(filename: str, sep: str = '|') -> List[List[str]]:
    """ reads file, splits into lines and splits lines by 'sep' """
    assert (path.exists(filename))
    file_ = open(filename).read().splitlines()
    return list(map(lambda line: line.split(sep), file_))


def clean_ASpath(aspath: ASpath) -> ASpath:
    """ removes ASs if previous AS is identical """
    if not aspath:
        return []
    prev = aspath[0]
    return_path = [prev]
    for asn in aspath[1:]:
        if asn != prev:
            return_path.append(asn)
        prev = asn
    return return_path


def paral(function: Callable, iters: List[Iterable], num_cores=-1):
    """ compute function parallel with arguments in iters.
    function(iters[0][0],iters[0][1],...)"""
    return Parallel(n_jobs=num_cores)(delayed(function)(*its)
                                      for its in tqdm(zip(*iters),
                                                      total=len(iters[0]),
                                                      unit="jobs",
                                                      dynamic_ncols=True,
                                                      desc=function.__name__))


def link_on_path(link: ASlink, path: ASpath) -> bool:
    """returns whether link is present on path"""
    # assuming that each AS occurs only once on path
    assert len(path) == len(set(path)), "path shall not contain loops"
    # no links to themselves
    assert link[0] != link[1], "link can't have the same AS twice"
    # link always consists of two ASs
    assert len(link) == 2, "a link consists of exactly two ASs"

    # check if both ASs in link are present in path
    if link[0] in path and link[1] in path:
        if (path.index(link[0])) == len(path) - 1:
            # avoid index out of bounds
            return False
        else:
            return path[path.index(link[0]) + 1] == link[1]
    else:
        return False


def init_as_rel(as_rel: str):
    if not os.path.exists(as_rel):
        raise FileNotFoundError(
            "Could not find file.. Remember to use absolute paths")

    global as_rel_dict
    lines = open(as_rel).read().splitlines()
    lines = [line.split('|') for line in lines if '#' not in line]
    keys = [tuple(line[:2]) for line in lines]
    keys += list(map(lambda x: x[::-1], keys))
    v_map = {-1: 'p2c', 0: 'p2p'}
    values = [v_map[int(line[2])] for line in lines]
    values += list(map(lambda x: x[::-1], values))
    as_rel_dict = dict(zip(keys, values))


def get_relationship(link: ASlink):
    assert as_rel_dict is not None, "init_as_rel first"
    try:
        return as_rel_dict[link]
    except KeyError:
        return "None"


def init_as_rank(as_rank_file: str):
    if not os.path.exists(as_rank_file):
        raise FileNotFoundError(
            "Could not find file.. Remember to use absolute paths")
    global as_rank_dict
    as_rank_dict = defaultdict(
        lambda: None,
        dict(map(lambda x: (int(x[0]), int(x[1])), rsp(as_rank_file,
                                                       sep=':'))))


def get_as_rank(asn: ASN) -> int:
    return as_rank_dict[int(asn)]


def init_country_codes(resource_dir: str):
    files = [
        "delegated-afrinic-extended-latest", "delegated-arin-extended-latest",
        "delegated-ripencc-extended-latest", "delegated-apnic-extended-latest",
        "delegated-lacnic-extended-latest"
    ]
    rirs = ["afrinic", "arin", "ripencc", "apnic", "lacnic"]
    global as_country_codes_dict
    as_country_codes_dict = defaultdict(lambda: None)
    for filename, rir in zip(files, rirs):
        lines = open(resource_dir + filename).read().splitlines()
        lines = [line.split('|') for line in lines if '#' not in line]
        for line in lines:
            if line[0] == rir and line[1] != '*' and line[3] != '*' and line[
                    2] == 'asn':
                as_country_codes_dict[int(line[3])] = line[1]


def get_country_code(asn: ASN) -> str:
    return as_country_codes_dict[int(asn)]


def get_cdf_space(data):
    return (sorted(data), 1. * np.arange(len(data)) / (len(data) - 1))


def prep_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def log(message: str):
    print(str(dt.now()) + "\t| " + message)


assert (get_AS_links_single([1, 2, 3, 4]) == [(1, 2), (2, 3), (3, 4)])
assert (get_AS_links_single([3, 4]) == [(3, 4)])
assert (get_AS_links_single([]) == [])
assert (get_AS_links_single([2]) == [])

assert (link_on_path((1, 2), [1, 2, 3, 4]))
assert (link_on_path((3, 4), [1, 2, 3, 4]))
assert (not link_on_path((4, 3), [1, 2, 3, 4]))
assert (not link_on_path((4, 5), [1, 2, 3, 4]))
assert (not link_on_path((1, 3), [1, 2, 3, 4]))
assert (not link_on_path((1, 3), []))
assert (not link_on_path((1, 3), [100]))

assert (clean_ASpath([1, 1, 2, 3, 3, 4]) == [1, 2, 3, 4])
assert (clean_ASpath([1, 1, 2, 3, 3, 4]) == [1, 2, 3, 4])
assert (clean_ASpath([1, 1, 2, 3, 3, 4]) == [1, 2, 3, 4])
assert (clean_ASpath([]) == [])
assert (clean_ASpath([1]) == [1])
assert (clean_ASpath([1, 1, 1, 1, 1]) == [1])
assert (clean_ASpath([1, 2, 3]) == [1, 2, 3])
assert (clean_ASpath([1, 2, 3, 4, 4, 4]) == [1, 2, 3, 4])

assert (get_AS_links([[1, 2, 3, 4]]) == {(1, 2), (2, 3), (3, 4)})
assert (get_AS_links([[1, 2]]) == {(1, 2)})
