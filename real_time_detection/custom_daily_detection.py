# daily_detection.py

import os
import json
import csv
import numpy as np
import torch
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from obspy import UTCDateTime

from utils.detect import smooth_moving_avg, detect_event_windows


import sys
sys.path.append('/home/ak287/seisbench/seisbench/models')
import seisbench.models as sbm


##
### 1. `daily_detection.py`

"""**Purpose:** Detect events at individual stations for a given time window.

**Workflow:**

1. **Load model & stations**:

   * Loads a pre-trained QuakeXNet model.
   * Loads a list of seismic stations from `stations.json`.
   * Defines a time window (default = past 24 hours), which can be customized.

2. **Download waveform data**:

   * Fetches continuous waveform data from IRIS for each station using ObsPy.

3. **Run model inference**:

   * Sliding window: **100 s length**, **10 s stride**.
   * Produces **class probabilities** for each window: `eq` (earthquake), `px` (explosion/phase), `su` (surface event), sampled every 10 s.

4. **Smooth probabilities**:

   * 5-sample moving average (~50 s) removes isolated spikes and short noise fluctuations.

5. **Detect events**:

   * Event **starts** when smoothed probability ≥ 0.15.
   * Event **ends** when smoothed probability < 0.15.
   * Event is only logged if **max probability ≥ 0.5** (default).
   * For each event, metrics are recorded: **mean probability**, **max probability**, **area under curve (AUC)**, start/end indices, and corresponding UTC timestamps.

**Output:**

* One CSV per station, with detected events and metrics. Example:

| station | network | class | auc  | mean_prob | max_prob | start_index | end_index | start_time           | end_time             |
| ------- | ------- | ----- | ---- | --------- | -------- | ----------- | --------- | -------------------- | -------------------- |
| PARA    | CC      | eq    | 3.37 | 0.35      | 0.54     | 5429        | 5438      | 2025-12-13T14:44:22Z | 2025-12-13T14:45:52Z |
| PARA    | CC      | eq    | 7.02 | 0.60      | 0.96     | 5561        | 5572      | 2025-12-13T15:06:22Z | 2025-12-13T15:08:12Z |

---"""

## running example - python custom_daily_detection.py  --start 2025-12-10T00:00:00  --end   2025-12-10T23:59:59


# -------------------- User Inputs --------------------
parser = argparse.ArgumentParser(description="Run QuakeXNet event detection on custom time range.")
parser.add_argument("--start", type=str, required=True,
                    help="Start time in UTC (e.g., '2025-12-10T00:00:00')")
parser.add_argument("--end", type=str, required=True,
                    help="End time in UTC (e.g., '2025-12-10T23:59:59')")
args = parser.parse_args()

st_time = UTCDateTime(args.start)
et_time = UTCDateTime(args.end)

print(f"Running detection from {st_time} to {et_time} ({et_time-st_time} seconds)")


# -------------------- Setup --------------------

# Load model
model = sbm.QuakeXNet.from_pretrained("base", version_str = '3')


# Load station list
with open("stations.json", "r") as f:
    stations = json.load(f)

# Create output folders
os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_file = "logs/detections.csv"

# Write header to CSV if not exists
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "station", "final_label", "eq_auc", "px_auc", "su_auc"])





# -------------------- Main Loop --------------------

client = Client("IRIS")
chn_prefix = "QuakeXNet_"
class_names = ['eq', 'px', 'su']
channel_map = {cls: f"{chn_prefix}{cls}" for cls in class_names}

for entry in stations:
    net = entry["net"]
    sta = entry["sta"]
    chn = entry.get("chn", "*H")

    print(f"🔍 Processing {net}.{sta}...")

    try:
        st = client.get_waveforms(
            network=net,
            station=sta,
            channel=chn + "*",
            location="*",
            starttime=st_time,
            endtime=et_time,
        )

        # Run inference
        probs_st = model.annotate(st, stride=500)

        total_auc = {}
        mean_probs = {}
        max_probs = {}
        start = {}
        end = {}

        for cls in class_names:
            total_auc[cls] = []
            mean_probs[cls] = []
            max_probs[cls] = []
            start[cls] = []
            end[cls] = []

            probs = probs_st.select(channel=channel_map[cls])

            for prob in probs:
                probs_array = np.array(prob)
                s_cls = smooth_moving_avg(probs_array)
                events = detect_event_windows(s_cls)

                for event in events:
                    total_auc[cls].append(event['area_under_curve'])
                    mean_probs[cls].append(event['mean_prob'])
                    max_probs[cls].append(event['max_prob'])
                    start[cls].append(event['start'])
                    end[cls].append(event['end'])
            
            
        
        event_records = []

        for cls in class_names:
            for i in range(len(total_auc[cls])):
                start_idx = start[cls][i]
                end_idx = end[cls][i]
                event_start_time = st_time + (start_idx * 500 / 50)  # stride=500, sampling=50Hz
                event_end_time = st_time + (end_idx * 500 / 50)

                event_records.append({
                    "station": sta,
                    "network": net,
                    "class": cls,
                    "auc": total_auc[cls][i],
                    "mean_prob": mean_probs[cls][i],
                    "max_prob": max_probs[cls][i],
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "start_time": str(event_start_time),
                    "end_time": str(event_end_time),
                })

        # Convert to DataFrame
        if event_records:
            df_events = pd.DataFrame(event_records)
            print(df_events)

            # Optionally save it to CSV
            
            # Format start and end times as YYYYMMDD_HHMM
            start_str = st_time.strftime("%Y%m%d_%H%M")
            end_str   = et_time.strftime("%Y%m%d_%H%M")

            # Save CSV with start and end times in filename
            df_events.to_csv(f"logs/{sta}_{start_str}_to_{end_str}_events.csv", index=False)

       



    except Exception as e:
        print(f"❌ Error processing {net}.{sta}: {e}")