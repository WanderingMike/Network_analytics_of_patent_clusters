import yake
from gensim.utils import simple_preprocess
import re
import gensim
import gensim.corpora as corpora
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import sys
import os


def print_output(type_script, process):
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

    # Remove punctuation and put to lowercase
    text = re.sub("():;-[,.!?]", '', text)
    text = text.lower()

    # R
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True) # yield returns without destroying state

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc))
                 if word not in stop_words] for doc in texts]

    data_words = list(sent_to_words(text))
    # remove stop words
    data_words = remove_stopwords(data_words)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # number of topics
    num_topics = 3
    
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    # Print the Keyword in the 3 topics
    doc_lda = lda_model[corpus]

    return doc_lda
