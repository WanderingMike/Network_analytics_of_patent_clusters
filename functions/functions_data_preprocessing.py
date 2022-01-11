import multiprocessing
from multiprocessing import Process
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import pickle

pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 25)
np.set_printoptions(threshold=sys.maxsize)


def fill_forward_citations(cluster, tensor_forward_citation, tensor_patent):
    years = 5
    period = timedelta(365*years)

    for index, row in cluster.iterrows():
        patent_date = tensor_patent[index]["date"]
        citedby_patent_num = 0
        for citedby_patent in tensor_forward_citation[index]:
            if citedby_patent["date"] - patent_date < period:
                citedby_patent_num += 1

        row["forward_citations"] = citedby_patent_num


def fill_cto(cluster):


def fill_pk(cluster, tensor_backward_citation):

    cluster["PK"] = cluster.index.apply(lambda patent_id: len(tensor_backward_citation[patent_id]))


def fill_sk(cluster, tensor_otherreference):

    cluster["SK"] = cluster.index.apply(lambda patent_id: tensor_otherreference[patent_id])


def fill_tct(cluster):


def fill_mf(cluster, tensor_patent_cpc):

    cluster["MF"] = cluster.index.apply(lambda patent_id: tensor_patent_cpc[patent_id]) # $ Main class or all classes? $


def fill_ts(cluster, tensor_patent_cpc):

    cluster["TS"] = cluster.index.apply(lambda patent_id: len(tensor_patent_cpc[patent_id]))


def fill_pcd(cluster, tensor_patent):

    cluster["PCD"] = cluster.index.apply(lambda patent_id: tensor_patent[patent_id]["num_claims"])


def fill_col(cluster, tensor_patent_assignee):

    cluster["COL"] = cluster.index.apply(lambda patent_id: 1 if len(tensor_patent_assignee[patent_id]) > 1 else 0)


def fill_inv(cluster, tensor_inventor):

    cluster["INV"] = cluster.index.apply(lambda patent_id: len(tensor_inventor[patent_id]))


def fill_tkh_ckh_tts_cts(cluster, tensor_patent_assignee, tensor_assignee_patent, tensor_cpc_patent, tensor_forward_citation, category):

    def search(patent_id):

        # Setting up variables
        total_know_how = 0
        core_know_how = 0
        total_strength = 0
        core_strength = 0

        assignee_list = tensor_patent_assignee[patent_id]

        # Looping through all patents
        for assignee in assignee_list:
            total_know_how += len(tensor_assignee_patent[assignee])

            for patent in tensor_assignee_patent[assignee]:  # $ do more efficient way? $
                if patent in tensor_cpc_patent[category]:
                    core_know_how += 1
                    core_strength += len(tensor_forward_citation[patent_id])
                total_strength += len(tensor_forward_citation[patent_id]) # $ risk of assignees collision? Create a set of patents?$

        return total_know_how, core_know_how, total_strength, core_strength


    cluster["TKH"], cluster["CKH"], cluster["TTS"], cluster["CTS"] = cluster.index.apply(lambda patent_id: search(patent_id))


def fill_pkh(cluster):

    cluster["PKH"] = cluster.apply(lambda row: row["TKH"] - row["CKH"])


def fill_pts(cluster):

    cluster["PTS"] = cluster.apply(lambda row: row["TTS"] - row["CTS"])


# maybe you can make it more efficient by calling assignees only once? Fewer functions basically...






