#!/bin/bash

zcat ../../data/march/all_all_updates_1583020800_1585699199_all_beacons.dump.gz ../../data/april/all_all_updates_1585785600_1588291199_all_beacons.dump.gz | pv | cut -d '|' -f 10,12 | egrep '45.132.188.0/24|147.28.32.0/24|147.28.36.0/24|147.28.40.0/24|147.28.44.0/24|147.28.48.0/24|147.28.52.0/24' | awk '!seen[$0]++' | gzip > beacon_paths.gz
