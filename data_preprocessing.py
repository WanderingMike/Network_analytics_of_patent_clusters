from functions.functions_data_preprocessing import *
import multiprocessing
from multiprocessing import Process
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer


def load_tensor(tensor_key):

    file = open("data/patentsview_cleaned/{}.pkl".format(tensor_key), "rb")
    tensor = pickle.load(file)

    return tensor


def generate_dataframe(tensor_cpc_patent, tensor_patent, category, start_date=None, end_date=None):

    patents_in_cpc_group = tensor_cpc_patent[category]
    indexed_patents = list()

    for patent in patents_in_cpc_group:
        try:
            if start_date <= tensor_patent[patent]["date"] <= end_date:
                indexed_patents.append(patent)
        except:
            pass

    cluster = pd.DataFrame(index=indexed_patents,
                           columns=['forward_citations', 'CTO', 'PK', 'SK', 'TCT', 'MF', 'TS',
                                    'PCD', 'COL', 'INV', 'TKH', 'CKH', 'TTS', 'CTS'])

    return cluster


def fill_dataframe(category, tensors, cluster):
    '''This function fills in every column of the final ML-readable dataframe. Each datapoint is sourced from the
    tensors given as parameters to the functions. Finally, missing values are replaced by the median for the each
    indicator/column in the dataframe.'''
    
    print("TKH, etc")
    cluster = fill_tkh_ckh_tts_cts(cluster, tensors["patent_assignee"], tensors["assignee_patent"], tensors["cpc_patent"], tensors["forward_citation"], category)
    print(cluster)

    cluster = fill_forward_citations(cluster, tensors["forward_citation"], tensors["patent"])
    print(cluster)

    cluster = fill_cto(cluster, tensors["patent_cpc"], tensors["backward_citation"])
    print(cluster)

    cluster = fill_pk(cluster, tensors["backward_citation"])
    print(cluster)

    cluster = fill_sk(cluster, tensors["otherreference"])
    print(cluster)

    cluster = fill_tct(cluster, tensors["backward_citation"], tensors["patent"])
    print(cluster)

    cluster = fill_mf_ts(cluster, tensors["patent_cpc"])
    print(cluster)

    cluster = fill_pcd(cluster, tensors["patent"])
    print(cluster)

    cluster = fill_col(cluster, tensors["patent_assignee"])
    print(cluster)

    cluster = fill_inv(cluster, tensors["inventor"])
    print(cluster)
    
    #cluster = fill_tkh_ckh_tts_cts(cluster, tensors["patent_assignee"], tensors["assignee_patent"], tensors["cpc_patent"],
    #                                tensors["forward_citation"], category)
    #print(cluster)



    # Encode mainclass column using one-hot-encoding
    cpc_to_labels = tensors["cpc_patent"].keys()
    le = LabelEncoder()
    le.fit(cpc_to_labels)
    cluster["MF"] = cluster["MF"].apply(lambda cpc_groups: le.transform(cpc_groups))

    ### OneHotEncoding
    mlb = MultiLabelBinarizer()
    onehotencoding = pd.DataFrame(mlb.fit_transform(cluster['MF']), columns=mlb.classes_, index=cluster.index)
    cluster = pd.concat([cluster, onehotencoding], axis=1, join="inner")
    cluster = cluster.drop(["MF"], axis=1)

    # Categorise output variables
    cluster["forward_citations"] = cluster["forward_citations"].apply(categorise_output, axis=1)

    return cluster

    #for column in ["TKH", "CKH", "PKH", "TTS", "CTS", "PTS"]:
    #    cluster[column] = cluster[column].replace(np.nan, cluster[column].median())


def data_preparation(categories, period_start, period_end):
    '''
    1) Load all tensors
    2) Create ML-readable dataframe
    3) Fill in that dataframe
    :param category: CPC group which interests us
    :return: ML-readable dataframe
    '''

    tensors = {
                "assignee": None,
                "cpc_patent": None,
                "patent_cpc": None,
                "otherreference": None,
                "patent": None,
                "patent_assignee": None,
                "assignee_patent": None,
                "inventor": None,
                "forward_citation": None,
                "backward_citation": None
    }

    #tensors["assignee_patent"] = load_tensor("assignee_patent")
    #tensors["patent_assignee"] = load_tensor("patent_assignee")
    #tensors["patent"] = load_tensor("patent")
    #tensors["forward_citation"] = load_tensor("forward_citation")
    print(datetime.now())
    for key in tensors.keys():
        tensors[key] = load_tensor(key)
    print(datetime.now())

    print("Tensors loaded.")
    for category in categories:
        cluster = generate_dataframe(tensors["cpc_patent"], tensors["patent"], category, period_start, period_end)
        print(cluster)
        print(datetime.now())
        return None
        cluster_complete = fill_dataframe(category, tensors, cluster)

    return cluster_complete

if __name__ == "__main__":
    categories = ["H04L"]
    period_start = datetime(2015, 1, 1)
    period_end = datetime(2015, 12, 31)
    df = data_preparation(categories, period_start, period_end)



