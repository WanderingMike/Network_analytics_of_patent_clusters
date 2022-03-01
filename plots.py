from main import *
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_pickle(name):
    ffile = open(name, "rb")
    loaded = pickle.load(ffile)
    ffile.close()
    return loaded


def save_pickle(name, data):
    ffile = open(name, "wb")
    pickle.dump(data, ffile)
    ffile.close()


### Load data
interesting_clusters = ['G06F16/9035',
                        'H04W12/55',
                        'Y10S707/915',
                        'G07B17/00508',
                        'G07B2017/00064',
                        'Y10S707/959',
                        'F41A17/20',
                        'H03K19/1731',
                        'H04W12/69',
                        'H04W12/60']

tensor_cpc_sub_patent = load_tensor("cpc_sub_patent")

df_final = pd.read_pickle("data/dataframes/df_final.pkl")
print(df_final)
cpc_time_series = load_pickle("data/clusters.pkl")
###


def time_series_data():

    series = {cpc_subgroup: dict() for cpc_subgroup in interesting_clusters}

    for cpc_subgroup in interesting_clusters:

        series[cpc_subgroup] = {year: None for year in
                                range(job_config.upload_date.year - 30, job_config.upload_date.year + 1)}

        patents_in_subgroup = tensor_cpc_sub_patent[cpc_subgroup]
        subgroup_df = df_final[df_final.index.isin(patents_in_subgroup)]

        for diff in range(31):
            year = job_config.upload_date.year - diff
            month = job_config.upload_date.month
            day = job_config.upload_date.day

            # Filtering patents
            start_date = subgroup_df["date"] > datetime(year - 1, month, day)
            end_date = subgroup_df["date"] <= datetime(year, month, day)

            filters = start_date & end_date
            temp_df = subgroup_df[filters]
            patents_per_year = list(temp_df.index.values)

            # Calculating indicators
            patent_count = len(patents_per_year)
            emergingness = temp_df["output"].mean()

            # Adding to time-series
            indicators = {"emergingness": emergingness,
                          "patent_count": patent_count}

            series[cpc_subgroup][year] = indicators

    return series


def time_series_plot(data):

    print("None")


def cdf_data():

    topical_clusters = df_final.index.tolist()
    emergingness_data = [cpc_time_series[group][job_config.upload_date.year] for group in topical_clusters]

    return emergingness_data


def cdf_plot(data):
    x_axis = np.sort(data)
    n = len(data)
    y_axis = np.array(range(n)) / float(n)

    plt.plot(x_axis, y_axis)
    plt.show()
    plt.savefig("output_tables/cumulative_density_distribution_03.png", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    ### Plot 1
    series = time_series_data()
    save_pickle("data/plots/time_series.pkl", data=series)
    time_series_plot(series)

    ### Plot 2
    emergingness_data = cdf_data()
    save_pickle("data/plots/emergingness_data.pkl", data=emergingness_data)
    cdf_plot(emergingness_data)

