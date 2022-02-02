import yake
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim.corpora as corpora
import nltk
import sys
import os
from sklearn.utils import resample
import pandas as pd
nltk.download('wordnet')
nltk.download('omw-1.4')

dataframe_length = 10000

def balance_dataset(df):
    df_majority = df[df.output==0]
    df_minority = df[df.output==1]
    length_output_1 = len(df_minority.index)

    if length_output_1 > dataframe_length:
        length_output_1 = dataframe_length

    df_majority_downsampled = resample(df_majority,
                                       replace=True,
                                       n_samples=length_output_1,
                                       random_state=123)

    df_upsampled = pd.concat([df_majority_downsampled, df_minority])

    print(df_upsampled.output.value_counts())
    
    return df_upsampled


def get_statistics(df):

    quantiles = df.forward_citations.quantile([0.25,0.5,0.75])
    print(quantiles)

    for count in [0,1,2,3,4,5,6,7,8,9]:
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

