from functions.functions_data_preprocessing import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

def load_tensors(tensors):
    for key in tensors.keys():
        file = open("data/patentsview_cleaned/{}.pkl".format(key), "rb")
        tensors[key] = pickle.load(file)

    return tensors


def generate_dataframe(tensor_cpc_patent, tensor_patent, category, start_date = None, end_date = None):

    patents_in_cpc_group = tensor_cpc_patent[category]
    indexed_patents = list()

    if start_date and end_date:
        for patent in patents_in_cpc_group:
            if start_date <= tensor_patent[patent]["date"] <= end_date:
                indexed_patents.append(patent)
    else:
        indexed_patents = patents_in_cpc_group

    cluster = pd.DataFrame(index = indexed_patents,
                           columns=['forward_citations', 'CTO', 'STO', 'PK', 'SK', 'TCT', 'MF', 'TS',
                                    'PCD', 'COL', 'INV', 'TKH', 'CKH', 'PKH', 'TTS', 'CTS', 'PTS'])

    return cluster


def fill_dataframe(category, tensors, cluster):
    '''This functions fills in every column of the final ML-readable dataframe. Each datapoint is sourced from the
    tensors given as parameters to the functions. Finally, missing values are replaced by the median for the each
    indicator/column in the dataframe.'''

    cluster = fill_forward_citations(cluster, tensors["forward_citation"], tensors["patent"])

    cluster = fill_cto(cluster, tensors["patent_cpc"], tensors["backward_citation"])

    cluster = fill_pk(cluster, tensors["backward_citation"])

    cluster = fill_sk(cluster, tensors["otherreference"])

    cluster = fill_tct(cluster, tensors["backward_citation"], tensors["patent"])

    cluster = fill_mf(cluster, tensors["patent_cpc"])

    cluster = fill_ts(cluster, tensors["patent_cpc"])

    cluster = fill_pcd(cluster, tensors["patent"])

    cluster = fill_col(cluster, tensors["patent_assignee"])

    cluster = fill_inv(cluster, tensors["inventor"])

    cluster = fill_tkh_ckh_tts_cts(cluster, tensors["patent_assignee"], tensors["assignee_patent"], tensors["cpc_patent"],
                                   tensors["forward_citation"], category)

    cluster = fill_pkh(cluster)

    cluster = fill_pts(cluster)

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


def data_preparation(category, period_start, period_end):
    '''
    1) Load all tensors
    2) Create ML-readable dataframe
    3) Fill in that dataframe
    :param category: CPC group which interests us
    :return: ML-readable dataframe
    '''

    tensors = {"assignee": None,
               "cpc_patent": None,
               "patent_cpc": None,
               "otherreference": None,
               "patent": None,
               "patent_assignee": None,
               "assignee_patent": None,
               "inventor": None,
               "forward_citation": None,
               "backward_citation": None}

    tensors = load_tensors(tensors)
    cluster = generate_dataframe(tensors["cpc_patent"], tensors["patent"], category, period_start, period_end)
    cluster_complete = fill_dataframe(category, tensors, cluster)

    return cluster_complete

if __name__ == "__main__":
    category = "H04W"
    period_start = datetime(2010, 1, 1)
    period_end = datetime(2015, 12, 31)
    data_preparation(category, period_start, period_end)



