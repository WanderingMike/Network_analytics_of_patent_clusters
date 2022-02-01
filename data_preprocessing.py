from functions.functions_data_preprocessing import *
import multiprocessing
from multiprocessing import Process
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer


def generate_dataframe(tensor_patent, start_date=None, end_date=None):

    indexed_patents = list()

    for patent in tensor_patent.keys():
        try:
            if start_date <= tensor_patent[patent]["date"] <= end_date:
                indexed_patents.append(patent)
        except:
            pass

    cluster = pd.DataFrame(index=indexed_patents,
                           columns=['date', 'forward_citations', 'CTO', 'PK', 'SK', 'TCT', 'MF', 'TS',
                                    'PCD', 'COL', 'INV', 'TKH', 'CKH', 'TTS', 'CTS'])

    return cluster


def fill_dataframe(tensors, cluster):
    '''This function fills in every column of the final ML-readable dataframe. Each datapoint is sourced from the
    tensors given as parameters to the functions. Finally, missing values are replaced by the median for the each
    indicator/column in the dataframe.'''
    
    print(datetime.now())
    cluster = fill_date_forward_citations(cluster, tensors["forward_citation"], tensors["patent"])

    print(datetime.now())
    cluster = fill_cto(cluster, tensors["patent_cpc_main"], tensors["backward_citation"])

    print(datetime.now())
    cluster = fill_pk(cluster, tensors["backward_citation"])

    print(datetime.now())
    cluster = fill_sk(cluster, tensors["otherreference"])

    print(datetime.now())
    cluster = fill_tct(cluster, tensors["backward_citation"], tensors["patent"])

    print(datetime.now())
    cluster = fill_mf_ts(cluster, tensors["patent_cpc_main"])

    print(datetime.now())
    cluster = fill_pcd(cluster, tensors["patent"])

    print(datetime.now())
    cluster = fill_col(cluster, tensors["patent_assignee"])

    print(datetime.now())
    cluster = fill_inv(cluster, tensors["inventor"])
    
    cluster = fill_tkh_ckh_tts_cts(cluster, tensors["patent_assignee"], tensors["assignee_patent"], tensors["patent_cpc_main"], tensors["forward_citation"])
    print(datetime.now())

    # Encode mainclass column using one-hot-encoding
    cpc_mainclass_labels = list(set[name[:4] for name in tensors["cpc_sub_patent"].keys()]))
    le = LabelEncoder()
    le.fit(cpc_mainclass_labels)
    cluster["MF"] = cluster["MF"].apply(lambda cpc_groups: le.transform(cpc_groups))

    ### OneHotEncoding
    mlb = MultiLabelBinarizer()
    onehotencoding = pd.DataFrame(mlb.fit_transform(cluster['MF']), columns=mlb.classes_, index=cluster.index)
    cluster = pd.concat([cluster, onehotencoding], axis=1, join="inner")
    cluster = cluster.drop(["MF"], axis=1)

    #cluster.to_csv("data/dataframes/test2.csv")

    return cluster

    #for column in ["TKH", "CKH", "PKH", "TTS", "CTS", "PTS"]:
    #    cluster[column] = cluster[column].replace(np.nan, cluster[column].median())


def data_preparation(tensors, period_start, period_end):
    '''
    1) Load all tensors
    2) Create ML-readable dataframe
    3) Fill in that dataframe
    :param category: CPC group which interests us
    :param tensors: full tensors which carry the necessary data
    :return: ML-readable dataframe
    '''

    cluster = generate_dataframe(tensors["patent"], period_start, period_end)
    print(datetime.now())
    cluster_complete = fill_dataframe(tensors, cluster)
    print(cluster)
    print(datetime.now())

    return cluster_complete



