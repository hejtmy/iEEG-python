import mne

def def_picks(raw):
    picks = mne.pick_types(raw.info, seeg = True, meg = True, eeg=False, stim=False, eog=True, exclude='bads')
    return picks