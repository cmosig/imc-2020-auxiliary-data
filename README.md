# Download of BGP update dumps

Run the python download script from the respective project directory which is
either `data/march` or `data/april`. Pass the configuration file as parameter.
The configuration file contains all information of the experiment such as
update interval or prefixes. 

    python3 ../../code/download_data.py config.ini

The script queries the Isolario, RIPE RIS, and RouteViews dump archives. Note
that you are required to download and install CAIDA's `bgpreader`, specifically
our custom fork:

    https://github.com/cmosig/libbgpstream   

# Preparation 

## Expected Updates

Run the script:

    python3 ../../code/expected_updates.py config.ini

This takes the crontabs from the config file for each prefix and converts them
into a list of updates with timestamps, which note where we send updates from
Beacons and thus expect them so see them at route collectors. The results of
this script are automatically saved.

# Path Labeling
