import yake
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim.corpora as corpora
import nltk
from sklearn.preprocessing import MultiLabelBinarizer
import sys
import os
from sklearn.utils import resample
import pandas as pd
from datetime import datetime
nltk.download('wordnet')
nltk.download('omw-1.4')

dataframe_length = 3000

def balance_dataset(df):

    print("2.2.1.0.1 Balancing dataset ({})".format(datetime.now()))
    df_majority = df[df.output==0]
    df_minority = df[df.output==1]
    length_output_1 = len(df_minority.index)

    if length_output_1 > dataframe_length:
        length_output_1 = dataframe_length

    df_majority_downsampled = resample(df_majority,
                                       replace=True,
                                       n_samples=length_output_1,
                                       random_state=123)

    df_minority_reduced = resample(df_minority,
                                   replace=True,
                                   n_samples=length_output_1,
                                   random_state=123)

    df_upsampled = pd.concat([df_majority_downsampled, df_minority_reduced])
    
    print("2.2.1.0.2 Output values for dataset ({})".format(datetime.now()))
    print(df_upsampled.output.value_counts())
    
    return df_upsampled


def get_statistics(df):

    quantiles = df.forward_citations.quantile([0.25, 0.5, 0.75])
    print(quantiles)

    for count in range(10):
        value = len(df[df["forward_citations"]==count].index)
        print("{}: {}".format(count, value)) 

    value_count_plus = len(df[df["forward_citations"]>=10].index)
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


def onehotencode(cluster, tensor_cpc_sub_patent, columns=None):

    cpc_mainclass_labels = list(set([name[:4] for name in tensor_cpc_sub_patent.keys()]))
    
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
    
    #for column in ["TKH", "CKH", "PKH", "TTS", "CTS", "PTS"]:
    #    cluster[column] = cluster[column].replace(np.nan, cluster[column].median())

   


