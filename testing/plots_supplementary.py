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
clusters_df = pd.read_csv("output_tables/clusters_df.csv")
print(clusters_df)
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

def real(df, ylim=(0, 1)):

    sns.set_theme(style="whitegrid")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 6))

    # Draw a violinplot with a narrower bandwidth than the default
    sns.violinplot(data=df, palette="Set3", bw=.2, cut=1, linewidth=1)

    # Finalize the figure
    ax.set(ylim=ylim)
    sns.despine(left=True, bottom=True)

    plt.show()


def create_time_series_df():
    cpc_time_series = load_pickle("data/clusters.pkl")
    topical_clusters = pd.read_csv("output_tables/clusters_df.csv")["CPC"].tolist()
    topical_clusters_time_series = {group: cpc_time_series[group] for group in topical_clusters}

    df_emergingness = pd.DataFrame(index=topical_clusters_time_series.keys(), columns=[2018, 2019, 2020, 2021])
    df_patent_count = pd.DataFrame(index=topical_clusters_time_series.keys(), columns=[2018, 2019, 2020, 2021])

    df_emergingness = fill_df(df_emergingness, "emergingness", topical_clusters_time_series)
    df_patent_count = fill_df(df_patent_count, "patent_count", topical_clusters_time_series)

    print(df_emergingness)
    print(df_patent_count)

    real(df_emergingness)
    real(df_patent_count, ylim=(0, np.percentile(df_patent_count.iloc[:, -1], 95)))


def fill_df(df, indicator, cpc_time_series):

    for year in df.columns:
        df[year] = df.index.map(lambda x: cpc_time_series[x][year][indicator])

    return df

create_time_series_df()

