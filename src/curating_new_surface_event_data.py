import numpy as np
import pandas as pd
import obspy
import json
from tqdm import tqdm





cat_new_su = pd.read_csv('../data/additional_surface_events_good_snr.csv', index_col = 0)
unique_start_times = np.unique(cat_new_su['trace_start_time'].values)
new_id = []
new_data = []
for i in tqdm(range(len(unique_start_times))):

    cat_temp = cat_new_su[cat_new_su['trace_start_time'] == unique_start_times[i]]  
    stns = cat_temp['station_code'].values
    nets = cat_temp['station_network_code'].values
    uids = cat_temp['event_id'].values
    
    unique_start_times[i] = str(obspy.UTCDateTime(unique_start_times[i]) - 70)
    
    t1 = unique_start_times[i].split('T')[0].replace('-','')
    t2 = unique_start_times[i].split('T')[1].split('.')[0].replace(':','')


    for j in range(len(stns)):
        try:
            a = obspy.read('../data/surface_event_waveforms/*'+stns[j]+'*'+t1+'*'+t2+'*')
            a.detrend()
            a.resample(100)
            if np.array(a).shape[-1] >= 18000:
                new_data.append(np.array(a)[:,0:18000])
                new_id.append(uids[j]+'_'+nets[j]+'.'+stns[j])
        except:
            pass

        
    
np.save('../data/new_curated_surface_event_data.npy', new_data)



# Save to a JSON file
with open("../data/new_curated_surface_event_ids.json", "w") as f:
    json.dump(new_id, f)