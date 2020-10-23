import pandas as pd
import configparser

# load config
config = configparser.ConfigParser()
config.read('config.ini')

pd.read_csv(config["general"]["input-file"],
            sep='|',
            header=0,
            names=["peer-ASn", "peer-IP"],
            usecols=[7, 8]).drop_duplicates().to_csv("asn_ip_mapping.csv",
                                                     sep="|",
                                                     index=False)
