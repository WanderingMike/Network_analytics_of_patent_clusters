from ML import *
from functions.functions_network_analysis import *
import networkx as nx
import pandas as pd

def prepare_overviews():

    return 


def technology_index(topical_clusters, cpc_time_series, tensors_cpc_sub_patent, end_year):
    '''
    :param topical_clusters: list of clusters to add to dataframe
    :param cpc_time_series: dictionary with emergingness and patent count per cluster
    '''
    cluster_descriptions = pd.read_csv("data/patentsview_data/cpc_subgroup.tsv", sep='\t', header=0, names=['CPC', 'desc'])

    clusters_df = pd.Dataframe(columns=['CPC', 'count', 'emergingness', 'delta(emergingness)'])
    
    # CPC
    clusters_df['CPC'] = topical_clusters
    # desc
    clusters_df = pd.merge(clusters_df, cluster_descriptions, how='left', left_on='CPC', right_on='CPC')
    # count
    clusters_df['count'] = clusters_df["CPC"].apply(lambda x: len(tensors_cpc_sub_patent[x]))
    # emergingness
    clusters_df['emergingness'] = clusters_df['CPC'].apply(lambda x: cpc_time_series[x][end_year]['emergingness'])
    # delta
    clusters_df['delta(emergingness)'] = clusters_df['CPC'].apply(lambda x: cpc_time_series[x][end_year]['emergingness'] - 
                                                                            cpc_time_series[x][end_year-1]['emergingness'])   
    # technology ind



    return clusters_df


def assignee_index():

    return


def impact_index():

    return


def norm_impact_index():

    return


def influence_index():

    ## Creating Graph
    print("6.4 Creating graph ({})".format(datetime.now()))
    network = nx.Graph()
    network.add_nodes_from(cpc_nodes)
    network.add_nodes_from(assignee_nodes)
    network.add_weighted_edges_from(edges)

    return

def unfold_network(cpc_time_series, tensors, topical_patents, end_year):
    ''' This function has four steps:
     1) Retrieving/calculating relevant data for all assignees, cpc patent clusters and patents themselves.
     2) Setting up network/graph
     3) Centrality analytics
     4) Further analytics and plots
    '''
    print("6.1 Finding topical clusters ({})".format(datetime.now())) 
    topical_clusters = find_topical_clusters(topical_patents, tensors["patent_cpc_sub"])

    print("6.2 Finding topical assignees ({})".format(datetime.now())) 
    topical_assignees = find_topical_assignees(topical_clusters, tensors["patent_assignee"], tensors["patent"])
   
    print(f'6.3 Preparing dataframes for both entities ({datetime.now()})')
    df_assignee, df_clusters = prepare_overviews()

    print("6.4 Getting nodes and edges ({})".format(datetime.now()))
    cpc_nodes = get_cpc_nodes(topical_clusters, cpc_time_series)
    assignee_nodes = get_assignee_nodes(topical_assignees)
    edges = fetch_edge_data(topical_assignees)
   
    # Indices
    print(f'6.5 Calculating Technology Index ({datetime.now()})')
    technology_index(topical_clusters, cpc_time_series, tensors["cpc_sub_patent"], end_year)
    print(f'6.6 Calculating Assignee Index ({datetime.now()})')
    assignee_index()
    print(f'6.7 Calculating Impact Index ({datetime.now()})')
    impact_index()
    print(f'6.8 Calculating Normalised Impact Index ({datetime.now()})')
    norm_impact_index()
    print(f'6.9 Calculating Influence Index ({datetime.now()})')
    influence_index()
    
    # Output
    print(f'6.10 Writing to output files ({datetime.now()})')

    
