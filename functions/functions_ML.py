import yake
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim.corpora as corpora
import nltk
import sys
import os
nltk.download('wordnet')
nltk.download('omw-1.4')


def print_output(type_script, process):
    '''
    Write process-level print functions to independent stdout and stderr files.
    :param type_script: head of ame of stdout and stderr files
    :param process: process ID
    '''
    print(os.getpid())
    sys.stdout = open("std_out/process/{}_{}.out".format(type_script, process), "w")
    sys.stderr = open("std_out/process/{}_{}.err".format(type_script, process), "w")


def categorise_output(citations):
    '''This functions categorises the ML-readable output column forward citations'''

    if citations >= 20:
        return 3
    elif 10 <= citations <= 19:
        return 2
    elif 2 <= citations <= 9:
        return 1
    elif 0 <= citations <= 1: 
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
    print(processed_docs)

    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                           num_topics=3,
                                           id2word=dictionary,
                                           passes=10,
                                           workers=2)
    # Print the Keyword in the 3 topics
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic))
        print("\n")

    return lda_model

