import pandas as pd
from datetime import datetime
from functions.config import job_config
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def load_pickle(name):
    """
    This function loads a pickle file
    :param name: name of file
    """

    ffile = open(name, "rb")
    loaded = pickle.load(ffile)
    ffile.close()
    return loaded


def save_pickle(name, data):
    """
    This function dumps the object 'name' in a pickle file
    :param name: name of file to be written to
    :param data: data object to dump
    """

    ffile = open(name, "wb")
    pickle.dump(data, ffile)
    ffile.close()


def cdf_data(cpc_time_series, topical_clusters):
    """Fetches topical clusters and the amount of patents per cluster per year"""

    emergingness_data = [cpc_time_series[group][job_config.data_upload_date.year] for group in topical_clusters]

    return emergingness_data


def prepare_violin_plot_df(cpc_time_series, topical_clusters, indicator):
    """Prepares violin plots datasets"""

    ### Prepare rows of data
    rows_list = []
    for group in cpc_time_series.keys():
        for year in range(2018, 2022):
            if group in topical_clusters:
                dict1 = {"name": group,
                         "year": year,
                         "type": "query",
                         "cluster value": cpc_time_series[group][year][indicator]}
            else:
                dict1 = {"name": group,
                         "year": year,
                         "type": "full",
                         "cluster value": cpc_time_series[group][year][indicator]}
            rows_list.append(dict1)

    ### Create pandas dataframe out of list of rows
    df = pd.DataFrame(rows_list)

    return df
