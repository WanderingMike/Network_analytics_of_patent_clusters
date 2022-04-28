from datetime import datetime, timedelta
from autosklearn import classification
import pickle
from tqdm import tqdm
import pickle
import networkx as nx
import pandas as pd
import math
import string
import sys
from sklearn.preprocessing import MultiLabelBinarizer
import os
from sklearn.utils import resample
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import multiprocessing
from multiprocessing import Process
import numpy as np
from dateutil.relativedelta import relativedelta
from statistics import median
from random import shuffle

pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 25)
np.set_printoptions(threshold=sys.maxsize)


def load_pickle(name):
    ffile = open(name, "rb")
    loaded = pickle.load(ffile)
    ffile.close()
    return loaded


def save_pickle(name, data):
    ffile = open(name, "wb")
    pickle.dump(data, ffile)
    ffile.close()


class MlConfig:

    def __init__(self):
        self.number_of_cores = 6
        self.search_min = 0
        self.search_hours = 6
        self.ml_search_time = self.search_min * 60 + self.search_hours * 3600
        self.size_dataframe_train = 15000
        self.data_upload_date = datetime(2021, 6, 30)
        self.prediction_timeframe_years = 5
        self.load_network = False
        self.load_topical_patents = True
        self.load_main = True
        self.load_df_final = True
        self.load_classifier = True
        self.load_df_filled = True
        self.graph_name = "cybersecurity_graph"
        self.keyphrases = [['allowlist',
                            'antimalware',
                            'antispyware',
                            'antivirus',
                            'asymmetric key',
                            'attack signature',
                            'blocklist',
                            'blue team',
                            'bot',
                            'botnet',
                            'bug',
                            'ciphertext',
                            'computer forensics',
                            'computer security incident',
                            'computer virus',
                            'computer worm',
                            'cryptanalysis',
                            'cryptography',
                            'cryptographic',
                            'cryptology',
                            'cyber incident',
                            'cybersecurity',
                            'cyber security',
                            'cyberspace',
                            'cyber threat intelligence',
                            'data breach',
                            'data leakage',
                            'data theft',
                            'decrypt',
                            'decrypted',
                            'decryption',
                            'denial of service',
                            'digital forensics',
                            'digital signature',
                            'encrypt',
                            'encrypted',
                            'encryption',
                            'firewall',
                            'hacker',
                            'hashing',
                            'keylogger',
                            'malware',
                            'malicious code',
                            'network resilience',
                            'password',
                            'pen test',
                            'pentest',
                            'phishing',
                            'private key',
                            'public key',
                            'red team',
                            'rootkit',
                            'spoofing',
                            'spyware',
                            'symmetric key',
                            'systems security analysis',
                            'threat actor',
                            'trojan',
                            'white team']]


job_config = MlConfig()

tensors_config = {"assignee": {"tensor": None,
                               "dataset": "assignee",
                               "leading_column": "assignee_id",
                               "remaining_columns": ["organisation"],
                               "tensor_value_format": None},
                  "cpc_sub_patent": {"tensor": None,
                                     "dataset": "cpc_current",
                                     "leading_column": "cpc_subgroup",
                                     "remaining_columns": ["patent_id"],
                                     "tensor_value_format": list()},
                  "patent_cpc_main": {"tensor": None,
                                      "dataset": "cpc_current",
                                      "leading_column": "patent_id",
                                      "remaining_columns": ["cpc_group"],
                                      "tensor_value_format": list()},
                  "patent_cpc_sub": {"tensor": None,
                                     "dataset": "cpc_current",
                                     "leading_column": "patent_id",
                                     "remaining_columns": ["cpc_subgroup"],
                                     "tensor_value_format": list()},
                  "otherreference": {"tensor": None,
                                     "dataset": "otherreference",
                                     "leading_column": "patent_id",
                                     "remaining_columns": ["otherreference"],
                                     "tensor_value_format": None},
                  "patent": {"tensor": None,
                             "dataset": "patent",
                             "leading_column": "patent_id",
                             "remaining_columns": ["date", "abstract", "num_claims"],
                             "tensor_value_format": dict()},
                  "patent_assignee": {"tensor": None,
                                      "dataset": "patent_assignee",
                                      "leading_column": "patent_id",
                                      "remaining_columns": ["assignee_id"],
                                      "tensor_value_format": list()},
                  "assignee_patent": {"tensor": None,
                                      "dataset": "patent_assignee",
                                      "leading_column": "assignee_id",
                                      "remaining_columns": ["patent_id"],
                                      "tensor_value_format": list()},
                  "inventor": {"tensor": None,
                               "dataset": "patent_inventor",
                               "leading_column": "patent_id",
                               "remaining_columns": ["inventors"],
                               "tensor_value_format": None},
                  "forward_citation": {"tensor": None,
                                       "dataset": "uspatentcitation",
                                       "leading_column": "citation_id",
                                       "remaining_columns": ["patent_id"],
                                       "tensor_value_format": list()},
                  "backward_citation": {"tensor": None,
                                        "dataset": "uspatentcitation",
                                        "leading_column": "patent_id",
                                        "remaining_columns": ["citation_id"],
                                        "tensor_value_format": list()},
                  "year_patent": {"tensor": None,
                                  "dataset": "patent",
                                  "leading_column": "year",
                                  "remaining_columns": ["patent_id"],
                                  "tensor_value_format": list()}
                  }

def show_value(predictor, to_print):
    if predictor:
        print(to_print)
