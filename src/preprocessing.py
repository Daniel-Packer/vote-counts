import pandas as pd


def preprocess(raw_data, drop_writein=True):
    data = raw_data.copy()
    data = data[~data["writein"]] if drop_writein else data
    data = data.replace("OVER VOTES", "OVERVOTES")
    data = data.replace("UNDER VOTES", "UNDERVOTES")
    return data
