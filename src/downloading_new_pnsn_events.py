import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import obspy
from tqdm import tqdm
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader



## Reading the catalog
# change this catalog based on what you want to download. 
df_exp = pd.read_csv('pnw_explosion_2023_2025.csv')


# extracting certain parameters
starttimes  =  df_exp['time'].values
ev_id = df_exp.index.values
ev_lats = df_exp['latitude'].values
ev_lons = df_exp['longitude'].values



for i in tqdm(range(len(df_exp))):
    
    try:

        origin_time = obspy.UTCDateTime(starttimes[i])

        # Circular domain around the epicenter. This will download all data between
        # 70 and 90 degrees distance from the epicenter. This module also offers
        # rectangular and global domains. More complex domains can be defined by
        # inheriting from the Domain class.
        domain = CircularDomain(latitude=ev_lats[i], longitude=ev_lons[i],
                                minradius=0, maxradius=0.5)

        restrictions = Restrictions(
            # Get data from 5 minutes before the event to one hour after the
            # event. This defines the temporal bounds of the waveform data.
            starttime=origin_time - 70,
            endtime=origin_time + 200,
            # You might not want to deal with gaps in the data. If this setting is
            # True, any trace with a gap/overlap will be discarded.
            reject_channels_with_gaps=True,
            # And you might only want waveforms that have data for at least 95 % of
            # the requested time span. Any trace that is shorter than 95 % of the
            # desired total duration will be discarded.
            minimum_length=1.0,
            # No two stations should be closer than 10 km to each other. This is
            # useful to for example filter out stations that are part of different
            # networks but at the same physical station. Settings this option to
            # zero or None will disable that filtering.
            minimum_interstation_distance_in_m= 5E3,
            # Only HH or BH channels. If a station has HH channels, those will be
            # downloaded, otherwise the BH. Nothing will be downloaded if it has
            # neither. You can add more/less patterns if you like.
            channel_priorities=["BH[ZNE]", "HH[ZNE]", "EH[ZNE]"],
            # Location codes are arbitrary and there is no rule as to which
            # location is best. Same logic as for the previous setting.
            location_priorities=["", "00", "10"])

        # No specified providers will result in all known ones being queried.
        mdl = MassDownloader(providers = ['IRIS'])
        # The data will be downloaded to the ``./waveforms/`` and ``./stations/``
        # folders with automatically chosen file names.
        mdl.download(domain, restrictions, mseed_storage="../data/pnw_new_explosion_2023_2025/waveforms/"+str(ev_id[i]),
                     stationxml_storage="../data/pnw_new_explosion/stations/"+str(ev_id[i]))
        
    except:
        pass
