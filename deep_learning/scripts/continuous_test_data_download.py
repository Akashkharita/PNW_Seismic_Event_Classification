#!/usr/bin/env python3
"""
Classify a month of continuous data for several stations in 100-second windows.
Outputs one CSV per station with the class probabilities per window.

Assumptions
-----------
* model2.annotate(Stream) → Stream with 4 Traces named
    SeismicCNN_eq/px/no/su (1 sample each = probability).
* Model expects 3-component, 50 Hz, 100 s Stream (5 000 samples/component).
* SeisBench or other preprocessing is handled **inside** `model2.annotate`
  or is not needed.  If you need custom preprocessing, insert it where marked.
"""

import os
import sys
import math
import csv
from datetime import timedelta

import obspy
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from tqdm import tqdm   # progress bar

import sys
sys.path.append('/home/ak287/seisbench/seisbench/models')
import seisbench.models as sbm


#model1 = sbm.QuakeXNetoneD.from_pretrained("base")
model2 = sbm.SeismicCNN.from_pretrained("base")






starttime = UTCDateTime(2023, 6, 1, 0, 0, 0)

stations = [
    {"network": "UW", "station": "RER",  "channel": "HH?"},
    {"network": "UW", "station": "STAR", "channel": "EH?"},
    {"network": "CC", "station": "PANH", "channel": "BH?"},
    {"network": "CC", "station": "OBSR", "channel": "BH?"},
    {"network": "CC", "station": "MILD", "channel": "BH?"},
]

window_sec = 100
stride_sec = 50
duration_sec =  24*3600*60 # One full day (86400 seconds)
num_windows = duration_sec // stride_sec

for sta in stations:
    output_path = f"../dl_probs_csv/{sta['network']}_{sta['station']}_{starttime.date}.csv"
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utc_start", "eq_prob", "px_prob", "no_prob", "su_prob"])  # header

        i = 0
        with tqdm(total=num_windows, desc=f"{sta['network']}.{sta['station']}", unit="win") as pbar:
            while i + window_sec <= duration_sec:
                try:
                    stream = client.get_waveforms(
                        network=sta["network"],
                        station=sta["station"],
                        channel=sta["channel"],
                        location="*",
                        starttime=starttime + i,
                        endtime=starttime + i + window_sec
                    )

                    probs = model2.annotate(stream)
                    row = [str(starttime + i)] + [float(tr.data[0]) for tr in probs]
                    writer.writerow(row)

                except Exception as e:
                    tqdm.write(f"[{sta['station']}] Error at {starttime+i}: {e}")

                i += stride_sec
                pbar.update(1)

print("\n✅ Finished processing all stations.")
