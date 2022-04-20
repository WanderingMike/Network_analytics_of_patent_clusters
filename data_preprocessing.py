from functions.functions_data_preprocessing import *


def generate_dataframe(tensor_patent):
    """Generates ML-readable data format with required indicators as columns and patents as row indices"""

    full_patent_list = tensor_patent.keys()
    print("2.1.1.1 Number of patents: {} ({})".format(len(full_patent_list), datetime.now()))

    cluster = pd.DataFrame(index=full_patent_list,
                           columns=['date', 'forward_citations', 'CTO', 'PK', 'TCS', 'SK', 'TCT', 'MF', 'TS',
                                    'PCD', 'COL', 'INV', 'TKH', 'CKH', 'TTS', 'CTS'])

    print("2.1.1.2 Running on {} patents ({})".format(len(cluster.index), datetime.now()))

    return cluster


def fill_dataframe(tensors, cluster):
    """This function fills in every column of the final ML-readable dataframe. Each datapoint is sourced from the
    tensors given as parameters to the functions. Finally, missing values are replaced by the median for the each
    indicator/column in the dataframe."""
    
    print("2.1.2.1 Calculating forward citations ({})".format(datetime.now()))
    cluster["date"], cluster["forward_citations"] = zip(*cluster.apply(
        lambda x: fill_date_forward_citations(x, tensors["forward_citation"], tensors["patent"]), axis=1))
    df_citations = cluster["forward_citations"]

    print("2.1.2.2 Calculating CTO ({})".format(datetime.now()))
    cluster["CTO"] = cluster.apply(lambda x: fill_cto(x, tensors["patent_cpc_main"], tensors["backward_citation"]), axis=1)

    print("2.1.2.3 Calculating PK/TCT/TCS ({})".format(datetime.now()))
    cluster["PK"], cluster["TCT"], cluster["TCS"] = zip(*cluster.apply(
        lambda x: fill_pk_tct_tcs(x, df_citations, tensors["backward_citation"], tensors["patent"]), axis=1))

    print("2.1.2.4 Calculating SK ({})".format(datetime.now()))
    cluster["SK"] = cluster.apply(lambda x: fill_sk(x, tensors["otherreference"]), axis=1)

    print("2.1.2.5 Calculating MF/TS ({})".format(datetime.now()))
    cluster["MF"], cluster["TS"] = zip(*cluster.apply(lambda x: fill_mf_ts(x, tensors["patent_cpc_main"]), axis=1))  # $ Main class or all classes? $

    print("2.1.2.6 Calculating PCD ({})".format(datetime.now()))
    cluster["PCD"] = cluster.apply(lambda x: fill_pcd(x, tensors["patent"]), axis=1)

    print("2.1.2.7 Calculating INV ({})".format(datetime.now()))
    cluster["INV"] = cluster.apply(lambda x: fill_inv(x, tensors["inventor"]), axis=1)
    
    print("2.1.2.8 Calculating TKH/CKH/TTS/CTS ({})".format(datetime.now()))
    cluster = fill_col_tkh_ckh_tts_cts(cluster,
                                       tensors["patent_assignee"],
                                       tensors["assignee_patent"],
                                       tensors["patent_cpc_main"],
                                       tensors["forward_citation"])
   
    print("2.1.2.9 Saving dataframe filled ({})".format(datetime.now()))
    print(cluster)
    cluster.to_pickle("data/dataframes/df_preprocessed.pkl")
    cluster.head(1000).to_csv("output_tables/df_preprocessed_head_1000.csv") # erase

    return cluster


def data_preprocessing(tensors):
    """
    1) Load all tensors
    2) Create ML-readable dataframe
    3) Fill in that dataframe
    :param tensors: full tensors which carry the necessary data
    :return: ML-readable dataframe
    """
    
    if job_config.load_df_filled:
        cluster_complete = pd.read_pickle("data/dataframes/df_preprocessed.pkl")
    else:
        print("2.1.1 Generating empty frame ({})".format(datetime.now()))
        cluster = generate_dataframe(tensors["patent"])
        print("2.1.2 Filling dataframe ({})".format(datetime.now()))
        cluster_complete = fill_dataframe(tensors, cluster)

    return cluster_complete
