from ML import *
from functions.functions_network_analysis import *
import networkx as nx
import pandas as pd
import math


def technology_index(topical_clusters, cpc_time_series, tensors_cpc_sub_patent, end_year):
    '''
    :param topical_clusters: list of clusters to add to dataframe
    :param cpc_time_series: dictionary with emergingness and patent count per cluster
    '''
    cluster_descriptions = pd.read_csv("data/patentsview_data/cpc_subgroup.tsv", sep='\t', header=0, names=['CPC', 'desc'])

    clusters_df = pd.Dataframe(columns=['CPC', 'count', 'emergingness', 'delta'])
    
    # CPC
    clusters_df['CPC'] = topical_clusters
    # desc
    clusters_df = pd.merge(clusters_df, cluster_descriptions, how='left', left_on='CPC', right_on='CPC')
    # count
    clusters_df['count'] = clusters_df["CPC"].apply(lambda x: len(tensors_cpc_sub_patent[x]))
    # emergingness
    clusters_df['emergingness'] = clusters_df['CPC'].apply(lambda x: cpc_time_series[x][end_year]['emergingness'])
    # delta
    clusters_df['delta'] = clusters_df['CPC'].apply(lambda x: cpc_time_series[x][end_year]['emergingness'] -
                                                              cpc_time_series[x][end_year-1]['emergingness'])

    def calculate_technology_index(cpc_subgroup):
        deltas_em = []
        deltas_pat = []

        for i in range(3):
            delta_em = cpc_time_series[cpc_subgroup][end_year-i]['emergingness'] - \
                       cpc_time_series[cpc_subgroup][end_year-i-1]['emergingness']

            delta_pat = cpc_time_series[cpc_subgroup][end_year-i]['patent_count'] - \
                        cpc_time_series[cpc_subgroup][end_year-i-1]['patent_count']

            norm_delta_pat = delta_pat / cpc_time_series[cpc_subgroup][end_year-i-1]['patent_count']
            deltas_em.append(delta_em)
            deltas_pat.append(norm_delta_pat)

        diff_emergingness = sum(deltas_em) / len(deltas_em)
        diff_patent_count = sum(deltas_pat) / len(deltas_pat)

        return diff_emergingness*diff_patent_count

    # technology index
    clusters_df['tech index'] = clusters_df['CPC'].apply(calculate_technology_index)

    return clusters_df


def assignee_index(topical_assignees, tensor_assignee):

    assignees_df = pd.Dataframe(columns=['ID', 'name', 'emergingness', 'count', 'impact', 'normalised impact', 'influence'])
    # ID
    assignees_df['ID'] = topical_assignees.keys()
    # name
    assignees_df['name'] = assignees_df['ID'].apply(lambda x: tensor_assignee[x]['organisation'])
    # emergingness
    assignees_df['emergingness'] = assignees_df['ID'].apply(lambda x: topical_assignees['emergingness'])

    return assignees_df


def impact_index(node, network):

    assignee_value = network[node]['weight']
    impact = 0
    count = 0
    for neighbour, value in network[node]:
        edge_weight = value['weight']
        count += edge_weight
        impact += assignee_value * edge_weight * network[neighbour]['weight']

    return impact, count


def network_indices(cpc_nodes, assignee_nodes, edges, assignee_df):

    # Creating Graph
    print(f'6.4 Creating graph ({datetime.now()})')
    network = nx.Graph()
    network.add_nodes_from(cpc_nodes)
    network.add_nodes_from(assignee_nodes)
    network.add_weighted_edges_from(edges)

    # impact
    for node in assignee_nodes:
        impact, length = impact_index(node, network)
        assignee_df[assignee_df['ID'] == node]['impact'] = impact
        assignee_df[assignee_df['ID'] == node]['normalised impact'] = impact/length

    phi = (1 + math.sqrt(5)) / 2.0  # largest eigenvalue of adj matrix
    centrality = nx.katz_centrality(network, 1 / phi - 0.01)
    for node, centrality_measure in sorted(centrality.items()):
        assignee_df[assignee_df['ID'] == node]['influence'] = centrality_measure

    return assignee_df


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

    print("6.4 Getting nodes and edges ({})".format(datetime.now()))
    cpc_nodes = get_cpc_nodes(topical_clusters, cpc_time_series)
    assignee_nodes = get_assignee_nodes(topical_assignees)
    edges = get_edge_data(topical_assignees)
   
    # Indices
    print(f'6.5 Calculating Technology Index ({datetime.now()})')
    clusters_df = technology_index(topical_clusters, cpc_time_series, tensors["cpc_sub_patent"], end_year)
    print(f'6.6 Calculating Assignee Index ({datetime.now()})')
    assignee_df = assignee_index(topical_assignees)
    print(f'6.7 Calculating Impact Index ({datetime.now()})')
    assignee_df = network_indices(cpc_nodes, assignee_nodes, edges, assignee_df)
    
    # Output
    print(f'6.10 Writing to output files ({datetime.now()})')
    print(clusters_df)
    print(assignee_df)

