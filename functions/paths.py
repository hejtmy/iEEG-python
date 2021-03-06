import os
from functions import read_eeg as readeeg


def participant_path(base, participant):
    return os.path.join(base, participant)


def unity_alloego_path(base, participant):
    return os.path.join(base, participant, 'UnityAlloEgo')


def eeg_path(experiment_path):
    return os.path.join(experiment_path, 'EEG', 'Preprocessed')


def bad_epochs_path(file_paths, append):
    """Creates path to a saved epoch file. 
    Used by mne_prepping write and read bad epochs functions

    Parameters
    ----------
    file_paths : list
        Paths generated by the prep_**_functions in this file
    append : str
        appendage to the name separating various versions of bad epochs

    Returns
    -------
    [type]
        [description]
    """
    fname = "bad-epochs" + append + ".txt"
    return os.path.join(file_paths['folder'], fname)


def tfr_epochs_path(file_paths, append):
    fname = "tfr-epochs" + append + "-tfr.h5"
    return os.path.join(file_paths['folder'], fname)


def prep_unity_alloego_files(base, participant):
    exp_path = eeg_path(unity_alloego_path(base, participant))
    files = dict()
    files['folder'] = exp_path
    files['montage'] = {
        'original': os.path.join(exp_path, participant + '_montage.csv'),
        'bipolar': os.path.join(
            exp_path, participant + '_montage_bipolar.csv'),
        'perElectrode': os.path.join(
            exp_path, participant + '_montage_perElectrode.csv'),
        'perHeadbox': os.path.join(
            exp_path, participant + '_montage_perHeadbox.csv'),
    }
    freq = str(readeeg.get_frequency(exp_path))
    files['EEG'] = {
        'original': os.path.join(exp_path, 'prep_' + freq + '.mat'),
        'bipolar': os.path.join(exp_path, 'prep_bipolar_' + freq + '.mat'),
        'perElectrode': os.path.join(exp_path, 'prep_perElectrode_' + freq + '.mat'),
        'perHeadbox': os.path.join(exp_path, 'prep_perHeadbox_' + freq + '.mat')
    }
    files['experiment'] = {
        'player': os.path.join(exp_path, participant + '_player.csv'),
        'events_eegtime': os.path.join(
            exp_path, participant + '_unity_eegtime_.csv'),
        'events_unitytime': os.path.join(
            exp_path, participant + '_unity_unitytime.csv'),
        'events_timesinceeegstart': os.path.join(
            exp_path, participant + '_unity_timesinceeegstart.csv'),
        'onsets': os.path.join(exp_path, participant + '_onsets.csv')
    }
    return files
