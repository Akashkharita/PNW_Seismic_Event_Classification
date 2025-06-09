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






# ────────────────────────── user config ──────────────────────────
START = UTCDateTime(2023, 6, 1)          # beginning of month (UTC)
END   = START + 60*24*3600           # ~30 days later
WINDOW_SEC   = 100                       # classifier input window
STRIDE_SEC = 50           # step between slice starts  ← NEW
CHUNK_SEC    = 3600                      # download one hour at a time
CLIENT       = Client("IRIS")            # or regional FDSN node
OUT_DIR      = "../dl_probs_csv"            # output folder
os.makedirs(OUT_DIR, exist_ok=True)

stations = [
    {"network": "UW", "station": "RER",  "channel": "HH?"},
    {"network": "UW", "station": "STAR", "channel": "EH?"},
    {"network": "CC", "station": "PANH", "channel": "BH?"},
    {"network": "CC", "station": "OBSR", "channel": "BH?"},
    {"network": "CC", "station": "MILD", "channel": "BH?"},
]

# import or create your classifier here
# from my_models import model2
# model2.eval()
# ──────────────────────────────────────────────────────────────────


def classify_window(win_st: Stream):
    """
    Returns dict of class:prob for one 100-s 3-component Stream.
    """
    out_st = model2.annotate(win_st)
    # Expect 4 traces in the output stream (eq, px, no, su)
    return {tr.stats.channel.split("_")[-1]: float(tr.data[0]) for tr in out_st}


def slice_hour_to_windows(hour_st: Stream, chunk_start: UTCDateTime):
    """
    Yield (win_start, win_stream) for consecutive 100-s windows with 50-s stride
    **inside the current 1-hour chunk**.

    We skip any window whose *end* would spill past the chunk boundary.
    The first window of the *next* chunk will cover that spill-over area,
    so we avoid duplicate work.
    """
    sr         = 50.0  # Hz  (adjust if model trained on a different rate)
    n_samples  = int(sr * WINDOW_SEC)
    last_start = CHUNK_SEC - WINDOW_SEC  # 3600-100 = 3500 s

    for offset in range(0, last_start + 1, STRIDE_SEC):
        win_start = chunk_start + offset
        win_end   = win_start   + WINDOW_SEC

        # Pull a copy, pad with zeros if a tiny gap exists
        win = hour_st.copy().trim(win_start, win_end,
                                  pad=True, fill_value=0, nearest_sample=False)

        # Require exactly one Z, one N, one E trace and right sample count
        if len(win) >= 3 and all(len(tr) == n_samples for tr in win):
            yield win_start, win



def process_station(sta):
    """
    Stream-through-time loop for one station. Writes a CSV incrementally.
    """
    out_path = os.path.join(OUT_DIR,
                  f"{sta['network']}_{sta['station']}_{START.date}.csv")
    header = ["utc_start", "eq_prob", "px_prob", "no_prob", "su_prob"]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        t = START
        total_hours = math.ceil((END - START) / CHUNK_SEC)
        with tqdm(total=total_hours,
                  desc=f"{sta['network']}.{sta['station']}",
                  unit="h") as pbar:
            while t < END:
                try:
                    hour_st = CLIENT.get_waveforms(
                        network   = sta["network"],
                        station   = sta["station"],
                        location  = "*",
                        channel   = sta["channel"],
                        starttime = t,
                        endtime   = t + CHUNK_SEC,
                        attach_response=False,
                    )
                    # OPTIONAL: pre-processing here (detrend, filter, …)

                    for win_start, win in slice_hour_to_windows(hour_st, t):
                        probs = classify_window(win)
                        writer.writerow([
                            win_start.isoformat(),
                            probs.get("eq", 0.0),
                            probs.get("px", 0.0),
                            probs.get("no", 0.0),
                            probs.get("su", 0.0)
                        ])

                except Exception as e:
                    # network hiccup or no data; skip hour
                    print(e)
                    pass

                t += CHUNK_SEC
                pbar.update(1)


# ───────────────────────────── main ──────────────────────────────
if __name__ == "__main__":
    for sta in stations:
        process_station(sta)

    print(f"\nFinished – CSVs saved in “{OUT_DIR}”.")
