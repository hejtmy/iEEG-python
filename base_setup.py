# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:02:25 2017

@author: hejtm
"""

# loading montage
pd_montage = readeegr.read_montage(path_montage) 
pd_montage_referenced  = readeegr.read_montage(path_montage_referenced) 

# Loading Unity data
#raw_original_vr = mneprep.load_raw(path_original_vr, FREQUENCY, pd_montage)
raw_perhead_vr = mneprep.load_raw(path_perhead_vr, FREQUENCY, pd_montage_referenced)
#raw_perelectrode_vr = mneprep.load_raw(path_perhead_vr, FREQUENCY, pd_montage_referenced)
#raw_bip_vr = mneprep.load_raw(path_bip_vr, FREQUENCY)

pd_unity_events = mneprep.load_unity_events(path_unity_events)
pd_matlab_events = mneprep.load_matlab_events(path_onset_events)
pd_events = pd.concat([pd_unity_events, pd_matlab_events])
pd_events = mneprep.clear_pd(pd_events)
mne_events_vr, mapp_vr = mneprep.pd_to_mne_events(pd_events, FREQUENCY)

## Epoching
epochs_perhead_vr = mne.Epochs(raw_perhead_vr, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3)
