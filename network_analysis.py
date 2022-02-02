from ML import *
from functions.functions_network_analysis import *
import networkx as nx


def calculate_centrality(network, cpc_time_series, assignee_time_series):

    return cpc_time_series, assignee_time_series


def prepare_plots(network):

    return None


def get_cpc_nodes(topical_clusters, cpc_time_series):

    cpc_nodes = list()

    for cluster in topical_clusters:

        cpc_info = (cluster, {"weight": cpc_times_series[cluster][2021]})
        cpc_nodes.append(cpc_info)
    
    return cpc_nodes


def get_assignee_nodes(topical_assignees):

    for assignee in topical_assignees.keys():

        assignee_info = (assignee, {"weight": topical_assignees[assignee]["emergingness"].mean()})
        assignee_nodes.append(assignee_info)

    return assignee_nodes


def get_edge_data(topical_assignees):

    edges = list()

    for assignee in assignees:

        for cluster in topical_assignees[assignee]:
            edge_info = (assignee, cluster, topical[assignee][cluster])
            edges.append(edge_info)

    return edges


def get_assignee_data(cluster, patent_value, topical_assignees):

    if assignee in topical_assignees:

        if cluster in topical_assignees[assignee]:
            topical_assignees[assignee][cluster] += 1
        else:
            topical_assignees[assignee][cluster] = 1

        topical_assignees[assignee]["emergingness"].append(patent_value)

    else:

        topical_assignees[assignee] = {"emergingness": [patent_value], cluster: 1}


def find_topical_assignees(topical_clusters, cpc_time_series, tensor_patent_assignee, tensor_patent):
    
    topical_assignees = dict()

    for cluster in topical_clusters:
        print("6.2.1 Finding topical assignees for cluster {} ({})".format(cluster, datetime.now()))

        for patent in cpc_time_series[cluster]["patents_final_year"]:

            try:
                assignees = tensor_patent_assignee[patent]
            except:
                continue

            try: 
                patent_value = tensor_patent[patent]["output"]
            except:
                patent_value = None

            for assignee in assignees:
                topical_assignees = get_assignee_data(cluster, patent_value, topical_assignees)

    return topical_assignees


def find_topical_clusters(topical_patents, tensor_patent_cpc_sub):
    
    cpc_subgroups = list()

    for patent in topical_patents:
        try:
            cpc_subgroups.append(tensor_patent_cpc_sub[patent])
        except:
            pass

    return list(set(cpc_subgroups))


def unfold_network(cpc_time_series, tensors, topical_patents):
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
   
    print("6.3 Getting nodes and edges ({})".format(datetime.now()))
    cpc_nodes = get_cpc_nodes(topical_clusters, cpc_time_series)
    assignee_nodes = get_assignee_nodes(topical_assignees)
    edges = fetch_edge_data(topical_assignees)

    # Creating Graph
    print("6.4 Creating graph ({})".format(datetime.now()))
    network = nx.Graph()
    network.add_nodes_from(cpc_nodes)
    network.add_nodes_from(assignee_nodes)
    network.add_weighted_edges_from(edges)

    # Centrality analytics
    print("6.5 Centrality analytics ({})".format(datetime.now()))
    # = calculate_centrality(network)



