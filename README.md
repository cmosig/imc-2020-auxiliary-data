# General Notes

We often say peer and vantage point (vp) are synonyms an refer the to peer of
route collectors. 

If you have questions don't hesitate to contact clemens.mosig@fu-berlin.de

Run the python scripts from the respective project directory which is
either `data/march` or `data/april`. Pass the configuration file as parameter.
The configuration file contains all information of the experiment such as
update interval or prefixes. 

# Preparation 

## Download of BGP update dumps

Run:

    python3 ../../code/download_data.py config.ini

The script queries the Isolario, RIPE RIS, and RouteViews dump archives. Note
that you are required to download and install CAIDA's `bgpreader`, specifically
our custom fork:

    https://github.com/cmosig/libbgpstream   

## Expected Updates

Run the script:

    python3 ../../code/expected_updates.py config.ini

This takes the crontabs from the config file for each prefix and converts them
into a list of updates with timestamps, which note where we send updates from
Beacons and thus expect them so see them at route collectors. The results of
this script are automatically saved.

## Create list of peers 

Simply run:

    python3 ../../code/generate_list_of_peers.py config.ini


## Match received updates with expected updates

Run the script:

    python3 ../../code/generate_missed_received_lists.py config.ini

This script uses loads the BGP update dumps, splits them by peer and then
matches the announcements with the Beacon events (generated from expected
updates file). Additionally, the matched updates are saved into a directory
(`missed_received_data`) split by peer IP. 


# Path Labeling

Run the script:

    python3 ../../code/label_paths.py config.ini

This script labels paths with RFD True and False based on the RFD signature.

# Pinpointing

## BeCAUSe 

Run: 

    python3 ../../code/BeCAUSe/path_format_BeCAUSe.py
    python3 ../../code/BeCAUSe/create_summaries.py


## Heuristics 

Run:

    python3 ../../code/graph_from_as_paths.py config.ini

This script creates an AS graph as seen from each vantage point. This
information is required for metric 2 when using the heuristics for pinpointing.

Then run the script:

    python3 ../../code/heuristics_pinpointing.py

