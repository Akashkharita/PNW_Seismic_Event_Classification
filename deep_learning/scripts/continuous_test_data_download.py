import obspy
import sys
sys.path.append('/home/ak287/seisbench/seisbench/models')
import seisbench.models as sbm
from obspy.clients.fdsn import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader, CircularDomain


client = Client('IRIS')


# List of specific stations
stations = [
    {"network": "UW", "station": "RCS", "channel": "EH?"},
    {"network": "UW", "station": "RER",  "channel": "HH?"},
    {"network": "UW", "station": "STAR",  "channel": "EH?"},
    {"network": "CC", "station": "PANH",  "channel": "BH?"},
    {"network": "CC", "station": "OBSR",  "channel": "BH?"}, 
    {"network": "CC", "station": "MILD",  "channel": "BH?"}, 
]

start = obspy.UTCDateTime(2023, 6, 1)
end   = obspy.UTCDateTime(2024, 1, 10)

# Create downloader
mdl = MassDownloader(providers=["IRIS"])

for sta in stations:
    print(f"⏳ Downloading {sta['network']}.{sta['station']}...")

    restrictions = Restrictions(
        starttime=start,
        endtime=end,
        chunklength_in_sec=86400,
        network=sta["network"],
        station=sta["station"],
        location= "*",              # Use "" or "*" to avoid missing data
        channel= sta["channel"],             # 3-component high-gain short-period
        reject_channels_with_gaps=True,  # <-- IMPORTANT: reject days with gaps
        minimum_length=0.99,             # Only download if >=99% of the chunk is available
        sanitize=True
    )

    # Use a dummy domain since station is fixed
    domain = CircularDomain(latitude=0, longitude=0, minradius=0, maxradius=180)

    mdl.download(
        domain,
        restrictions,
        mseed_storage=f"../test_data/waveforms/{sta['network']}.{sta['station']}",
        stationxml_storage=f"../test_data/stations/{sta['network']}.{sta['station']}"
    )
