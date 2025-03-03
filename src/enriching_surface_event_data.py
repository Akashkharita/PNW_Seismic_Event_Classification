import pandas as pd
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from tqdm import tqdm
import obspy
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader



client = Client("IRIS")



cat = pd.read_csv("/1-fnp/cascadia/c-whd01/yiyu_data/PNWML/exotic_metadata.csv")

# selecting surface events from catalog
cat_su = cat[cat['source_type'] == 'surface event']

# collecting unique event ids
uids = np.unique(cat_su['event_id'])


for i in tqdm(range(len(uids))):
    event_id = uids[i]
    lat = cat_su[cat_su['event_id'] == event_id].iloc[0]['station_latitude_deg']
    lon = cat_su[cat_su['event_id'] == event_id].iloc[0]['station_longitude_deg']
    starttime = UTCDateTime(cat_su[cat_su['event_id'] == event_id].iloc[0]['trace_start_time'])

    origin_time = starttime - 70

    # Circular domain around the epicenter. This will download all data between
    # 70 and 90 degrees distance from the epicenter. This module also offers
    # rectangular and global domains. More complex domains can be defined by
    # inheriting from the Domain class.
    domain = CircularDomain(latitude= lat, longitude= lon,
                            minradius=0, maxradius= 0.3)

    restrictions = Restrictions(
        # Get data from 5 minutes before the event to one hour after the
        # event. This defines the temporal bounds of the waveform data.
        starttime=origin_time,
        endtime=origin_time + 180,
        # You might not want to deal with gaps in the data. If this setting is
        # True, any trace with a gap/overlap will be discarded.
        reject_channels_with_gaps=True,
        # And you might only want waveforms that have data for at least 95 % of
        # the requested time span. Any trace that is shorter than 95 % of the
        # desired total duration will be discarded.
        minimum_length=0.95,
        # No two stations should be closer than 10 km to each other. This is
        # useful to for example filter out stations that are part of different
        # networks but at the same physical station. Settings this option to
        # zero or None will disable that filtering.
        minimum_interstation_distance_in_m=10E3,
        # Only HH or BH channels. If a station has HH channels, those will be
        # downloaded, otherwise the BH. Nothing will be downloaded if it has
        # neither. You can add more/less patterns if you like.
        channel_priorities=["HH[ZNE]", "BH[ZNE]"],
        # Location codes are arbitrary and there is no rule as to which
        # location is best. Same logic as for the previous setting.
        location_priorities=["", "00", "10"])

    # No specified providers will result in all known ones being queried.
    mdl = MassDownloader(providers = ['IRIS'])
    # The data will be downloaded to the ``./waveforms/`` and ``./stations/``
    # folders with automatically chosen file names.
    mdl.download(domain, restrictions, mseed_storage="../data/surface_event_waveforms",
                 stationxml_storage="../data/surface_event_stations")