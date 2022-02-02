from functions.functions_data_preprocessing import *
import multiprocessing
from multiprocessing import Process
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer


def generate_dataframe(tensor_patent, start_date=None, end_date=None):

    indexed_patents = list()
    full_patent_list = tensor_patent.keys()
    print("2.1.1.1 Number of patents: {} ({})".format(len(full_patent_list), datetime.now()))

    for patent in full_patent_list:
        try:
            if start_date <= tensor_patent[patent]["date"] <= end_date:
                indexed_patents.append(patent)
        except:
            pass

    cluster = pd.DataFrame(index=indexed_patents,
                           columns=['date', 'forward_citations', 'CTO', 'PK', 'SK', 'TCT', 'MF', 'TS',
                                    'PCD', 'COL', 'INV', 'TKH', 'CKH', 'TTS', 'CTS'])

    print("2.1.1.2 Running on {} patents ({})".format(len(cluster.index), datetime.now()))

    return cluster


def fill_dataframe(tensors, cluster):
    '''This function fills in every column of the final ML-readable dataframe. Each datapoint is sourced from the
    tensors given as parameters to the functions. Finally, missing values are replaced by the median for the each
    indicator/column in the dataframe.'''
    
    print("2.1.2.1 Calculating forward citations ({})".format(datetime.now()))
    cluster = fill_date_forward_citations(cluster, tensors["forward_citation"], tensors["patent"])

    print("2.1.2.2 Calculating CTO ({})".format(datetime.now()))
    cluster = fill_cto(cluster, tensors["patent_cpc_main"], tensors["backward_citation"])

    print("2.1.2.3 Calculating PK ({})".format(datetime.now()))
    cluster = fill_pk(cluster, tensors["backward_citation"])

    print("2.1.2.4 Calculating SK ({})".format(datetime.now()))
    cluster = fill_sk(cluster, tensors["otherreference"])

    print("2.1.2.5 Calculating TCT ({})".format(datetime.now()))
    cluster = fill_tct(cluster, tensors["backward_citation"], tensors["patent"])

    print("2.1.2.6 Calculating MF/TS ({})".format(datetime.now()))
    cluster = fill_mf_ts(cluster, tensors["patent_cpc_main"])

    print("2.1.2.7 Calculating PCD ({})".format(datetime.now()))
    cluster = fill_pcd(cluster, tensors["patent"])

    print("2.1.2.8 Calculating COL ({})".format(datetime.now()))
    cluster = fill_col(cluster, tensors["patent_assignee"])

    print("2.1.2.9 Calculating INV ({})".format(datetime.now()))
    cluster = fill_inv(cluster, tensors["inventor"])
    
    print("2.1.2.10 Calculating TKH/CKH/TTS/CTS ({})".format(datetime.now()))
    cluster = fill_tkh_ckh_tts_cts(cluster, tensors["patent_assignee"], tensors["assignee_patent"], tensors["patent_cpc_main"], tensors["forward_citation"])
    
    cluster.to_pickle("data/dataframes/filled_df.pkl")
    print(cluster)
    # Encode mainclass column using one-hot-encoding
    cpc_mainclass_labels = list(set([name[:4] for name in tensors["cpc_sub_patent"].keys()]))
    print(cpc_mainclass_labels)
    le = LabelEncoder()
    le.fit(cpc_mainclass_labels)
    cluster["MF"] = cluster["MF"].apply(lambda cpc_groups: le.transform(cpc_groups))

    ### OneHotEncoding
    mlb = MultiLabelBinarizer()
    onehotencoding = pd.DataFrame(mlb.fit_transform(cluster['MF']), columns=mlb.classes_, index=cluster.index)
    cluster = pd.concat([cluster, onehotencoding], axis=1, join="inner")
    cluster = cluster.drop(["MF"], axis=1)

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
    
    print("2.1.1 Generating empty frame ({})".format(datetime.now()))
    cluster = generate_dataframe(tensors["patent"], period_start, period_end)

    print("2.1.2 Filling dataframe ({})".format(datetime.now()))
    cluster_complete = fill_dataframe(tensors, cluster)
    print(cluster)

    cluster_complete.to_pickle("data/dataframes/ml_df.pkl")

    return cluster_complete



