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

def print_output(path):
    '''
    Write process-level print functions to independent stdout and stderr files.
    :param process: file path
    '''
    print(os.getpid())
    sys.stdout = open("{}.out".format(path), "w")
    sys.stderr = open("{}.err".format(path), "w")


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
    zeroes = len(df[df["forward_citations"]==0].index)
    ones = len(df[df["forward_citations"]==1].index)
    twos = len(df[df["forward_citations"]==2].index)
    threes = len(df[df["forward_citations"]==3].index)
    fours = len(df[df["forward_citations"]==4].index)
    fives = len(df[df["forward_citations"]==5].index)
    sixes = len(df[df["forward_citations"]==6].index)
    sevens = len(df[df["forward_citations"]==7].index)
    eights = len(df[df["forward_citations"]==8].index)
    nines = len(df[df["forward_citations"]==9].index)
    tens_above = len(df[df["forward_citations"]>=10].index)
    print("Zeroes: {}\nOnes: {}\nTwos: {}\nThrees: {}\nFours: {}\nFives: {}\nSixes: {}\nSevens: {}\nEights: {}\nNines: {}\nTens and above: {}".format(zeroes, ones, twos, threes, fours, fives, sixes, sevens, eights, nines, tens_above))

    return quantiles.loc[0.75]


def categorise_output(citations, median_value):
    '''This functions categorises the ML-readable output column forward citations'''

    if citations > median_value:
        return 1
    elif citations <= median_value:
        return 0
    else:
        return None


def extract_keywords(text):
    '''
    This function extracts keywords from the corpus of abstracts thanks to the spacy library.
    :param text: abstract corpus to extract topics from
    :return:
    '''

    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    num_keywords = 20
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                                top=num_keywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)

    return keywords


def extract_topic(text):
    '''
    This function extracts the main 3 topics from all patents in a CPC cluster in a given year. This topic modeling
    is done via the Latent Dirichlet Allocation (LDA) method.
    :param text: abstract corpus to extract topics from
    :return: list of 3 tuples
    '''

    # Word stemmer
    stemmer = SnowballStemmer("english")

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    # Tokenize and lemmatize
    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))

        return result

    words = []
    for word in text.split(' '):
        words.append(word)
    print("\n\nTokenized and lemmatized document: ")
    processed_docs = [preprocess(text)]
    print("Processed")

    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                           num_topics=1,
                                           id2word=dictionary,
                                           passes=10,
                                           workers=2)
    # Print the Keyword in the 3 topics
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic))
        print("\n")

    return lda_model

