def remove_unnamed(pd_dataframe):
    cols = [c for c in pd_dataframe.columns if 'Unnamed' not in c]
    return pd_dataframe[cols]