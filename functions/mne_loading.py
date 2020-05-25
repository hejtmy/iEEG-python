from functions import read_eeg as readeegr


def load_eeg(file_paths, type='original'):
    """[summary]

    Parameters
    ----------
    files_paths : dictionary
        obtained with prep_**_files functions from paths
    type : str, optional
        eeg type. Can be ['original', 'perHeadbox', 'perElectrode', 'bipolar']
        default "original"
    Returns
    -------
    mne.Raw, pandas.DataFrame
        mne.Raw eeg data and appropriate pandas DataFrame with montage
    """
    valid_types = ['original', 'perHeadbox', 'perElectrode', 'bipolar']
    if type not in valid_types:
        raise Exception('You have passed invalid type of eeg')
    frequency = readeegr.get_frequency(file_paths['folder'])
    montage = readeegr.read_montage(file_paths['montage'][montage_type(type)])
    eeg = load_raw(file_paths['EEG'][type], frequency, montage)
    return eeg, montage


def montage_type(eeg_type):
    """Returns montage type based on eeg type

    Parameters
    ----------
    eeg_type : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return {
        'original': 'original',
        'perHeadbox': 'referenced',
        'perElectrode': 'referenced',
        'bipolar': 'bipolar'
    }[eeg_type]


def load_raw(data_path, frequency, montage=None):
    """Loads eeg data from given filepath and converts to mne raw
    Parameters
    ----------
    data_path : str
        file path to the eeg mat file
    frequency : int
        frequnecy of the recording
    montage : pandas.DataFrame
        montage as loaded by read_montage

    Returns
    -------
    mne.raw eeg
    """
    data = readeegr.read_mat(data_path)
    raw = readeegr.eeg_mat_to_mne(data, frequency, montage)
    return raw
