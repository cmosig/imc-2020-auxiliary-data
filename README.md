### This is a copy of: https://git.rg.net/bgp-rfd/imc-aux

# General Notes

If you have questions don't hesitate to contact clemens.mosig@fu-berlin.de and
caitlin.gray@student.adelaide.edu.au.

Run the python scripts from the respective project directory which is either
data/march or data/april. The configuration file (config.ini) contains all
information of the experiment such as update interval or prefixes. 

Some scripts are very resource intensive, but scale well on machines with many
cores. If you want to test the scripts on your laptop reduce the measurement
period from one month to something smaller such as a day.

Peer and vantage point (vp) are synonyms an refer the to peer of route
collectors. 

# Run all

From the respective project directory run `../../code/run_all.sh`. Make sure
you have our fork of libbgpstream installed and all python packages in
requirements.txt.

# Preparation 

## Download of BGP update dumps

Run:

    python3 ../../code/download_data.py 

The script queries the Isolario, RIPE RIS, and RouteViews dump archives. Note
that you are required to download and install CAIDA's `bgpreader`, specifically
our custom fork:

    https://github.com/cmosig/libbgpstream   

## Expected Updates

Run the script:

    python3 ../../code/expected_updates.py 

This takes the crontabs from the config file for each prefix and converts them
into a list of updates with timestamps, which note where we send updates from
Beacons and thus expect them so see them at route collectors. The results of
this script are automatically saved.

## Create list of peers 

Simply run:

    python3 ../../code/generate_list_of_peers.py 

Create list of peers based on the update dump.

## Create ASN IP mapping

Simply run:

    python3 ../../code/generate_asn_vp_ip_mapping.py

Create a mapping between peer IP and AS number.


## Match received updates with expected updates

Run the script:

    python3 ../../code/generate_missed_received_lists.py 

This script uses loads the BGP update dumps, splits them by peer and then
matches the announcements with the Beacon events (generated from expected
updates file). Additionally, the matched updates are saved into a directory
(`missed_received_data`) split by peer IP. 


# Path Labeling

Run the script:

    python3 ../../code/label_paths.py 

This script labels paths with RFD True and False based on the RFD signature.

# Pinpointing

## BeCAUSe 

    python3 ../../code/path_format_BeCAUSe.py
    python3 ../../code/MCMC_network_tomography.py
    python3 ../../code/HMC_network_tomography.py
    python3 ../../code/create_summaries.py

## Heuristics 

Run:

    python3 ../../code/graph_from_as_paths.py 

This script creates an AS graph as seen from each vantage point. This
information is required for metric 2 when using the heuristics for pinpointing.

Then run the script:

    python3 ../../code/heuristics_pinpointing.py

This script pinpoints ASes based on the heuristic approach. 

# Plots

All plots from our paper are available in the /plots directory.
