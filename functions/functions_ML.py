from functions.config import *


def balance_dataset(df):
    '''Downsamples the binary output dataframe df in order to work with a balanced dataset'''

    print("2.2.1.0.1 Balancing dataset ({})".format(datetime.now()))
    df_majority = df[df.output == 0]
    df_minority = df[df.output == 1]
    length_output_1 = len(df_minority.index)

    if length_output_1 > job_config.size_dataframe_train:
        length_output_1 = job_config.size_dataframe_train

    df_majority_downsampled = resample(df_majority,
                                       replace=True,
                                       n_samples=length_output_1,
                                       random_state=123)

    df_minority_downsampled = resample(df_minority,
                                   replace=True,
                                   n_samples=length_output_1,
                                   random_state=123)

    df_balanced = pd.concat([df_majority_downsampled, df_minority_downsampled])
    
    print("2.2.1.0.2 Output values for dataset ({})".format(datetime.now()))
    print(df_balanced.output.value_counts())
    
    return df_balanced


def get_statistics(df):
    '''Statistics of df: quantiles, forward_citations distribution'''

    quantiles = df.forward_citations.quantile([0.25, 0.5, 0.75])
    print(quantiles)

    for count in range(10):
        value = len(df[df["forward_citations"] == count].index)
        print("{}: {}".format(count, value)) 

    value_count_plus = len(df[df["forward_citations"] >= 10].index)
    print("10+: ", value_count_plus)

    return quantiles.loc[0.75]


def categorise_output(citations, median_value):
    '''This functions categorises the ML-readable output column forward citations'''

    if citations > median_value:
        return 1
    elif citations <= median_value:
        return 0
    else:
        return None


def onehotencode(cluster, columns=None):
    '''OneHotEncoding of CPC subclass (MF) column'''

    # OneHotEncoding
    mlb = MultiLabelBinarizer()
    onehotencoding = pd.DataFrame(mlb.fit_transform(cluster['MF']), columns=mlb.classes_, index=cluster.index)
    cluster = pd.concat([cluster, onehotencoding], axis=1, join="inner")
    cluster = cluster.drop(["MF"], axis=1)

    if columns:
        cluster = cluster[cluster.columns.intersection(columns)]
        cols = cluster.columns.values
        print(cols, len(cols))
        return cluster, None

    else:
        cols = cluster.columns.values
        print(cols, len(cols))
        return cluster, cluster.columns.values.tolist()


   


