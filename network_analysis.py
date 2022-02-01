from ML import *
from functions.functions_network_analysis import *
import networkx as nx


def fetch_edge_data(tensors, cpc_time_series, assignee_series):
    '''
    Drawing all network edges for three sets of relationships:
    1) assignee-CPC
    2) CPC-CPC
    3) assignee-assignee
    :param network: network to add edges to
    :param tensors: data to create links
    :return: linked network
    '''

    assignee_list = list(assignee_series.keys())
    cpc_list = list(tensors["cpc_patent"].keys())
    edges = list()

    # assignee-CPC
    num_assignees = len(assignee_list)
    print("assignees: ", num_assignees)
    count = 0
    for assignee in assignee_list:
        count += 1
        if count % 10 == 0:
            print("Task 1: {}/{}".format(count, num_assignees))
        for cpc in cpc_list:
            try:
                weight = find_intersection(assignee_series[assignee],
                                           tensors["cpc_patent"][cpc])
                edges.append((assignee, cpc, weight))
            except Exception as e:
                print(e)


    # assignee-assignee
    count = 0
    for assignee1 in assignee_list:
        assignee_list.remove(assignee1)
        count += 1
        if count % 10 == 0:
            print("Task 2: {}/{}".format(count, num_assignees))
        for assignee2 in assignee_list:
            try:
                weight = find_intersection(assignee_series[assignee1],
                                           assignee_series[assignee2])
                edges.append((assignee1, assignee2, weight))

            except Exception as e:
                print(e)


    # CPC-CPC
    for cpc1 in cpc_list:
        cpc_list.remove(cpc1)
        for cpc2 in cpc_list:
            try:
                weight = find_intersection(tensors["cpc_patent"][cpc1],
                                           tensors["cpc_patent"][cpc2])
                edges.append((cpc1, cpc2, weight))

            except Exception as e:
                print(e)

    return edges


def update_edges(network, edges):
    '''Updating edge weights between two years'''

    for edge in edges:
        subject_id = edge[0]
        object_id = edge[1]
        weight = edge[2]
        network[subject_id][object_id]['weight'] += weight

    return network


def calculate_centrality(network, cpc_time_series, assignee_time_series):

    return cpc_time_series, assignee_time_series


def prepare_plots(network):

    return None


def find_topical_clusters(topical_patents, tensor_patent_cpc_sub):
    
    cpc_subgroups = list()

    for patent in topical_patents:
        try:
            cpc_subgroups.append(tensor_patent_cpc_sub[patent])
        except:
            pass

    return set(cpc_subgroups)


def get_assignee_data(cluster, topical_assignees):

    if assignee in topical_assignees:
        topical_assignees[assignee]
    else:
        topical_assignees[assignee] = [output]




def find_topical_assignees(topical_clusters, cpc_time_series, tensor_patent_assignee, tensor_patent):
    
    topical_assignees = dict()

    for cluster in topical_clusters:

        for patent in cpc_time_series[cluster]["patents_final_year"]:

            try:
                assignees = tensor_patent_assignee[patent]
            except:
                continue

            for assignee in assignees:

                try:
                    topical_assignees = get_assignee_data(cluster, topical_assignees)
                except:
                    pass

    return topical_assignees
    

def unfold_network(cpc_time_series, tensors, topical_patents):
    ''' This function has four steps:
     1) Retrieving/calculating relevant data for all assignees, cpc patent clusters and patents themselves.
     2) Setting up network/graph
     3) Centrality analytics
     4) Further analytics and plots
    '''
    
    topical_clusters = find_topical_clusters(topical_patents, tensors["patent_cpc_sub"])
    topical_assignees = find_topical_assignees(topical_clusters, tensors["patent_assignee"], tensors["patent"])


    # Creating Graph
    network = nx.Graph()
    network.add_nodes_from(cpc_time_series.keys())
    network.add_nodes_from(tensors["assignee_patent"].keys())

    print("Fetching edge data")
    print(datetime.now())
    edges = fetch_edge_data(tensors, cpc_time_series, assignee_series)
    print("Fetched edge data")
    print(datetime.now())

    # Draw edges
    network = update_edges(network, edges)

    # Centrality analytics
    #cpc_time_series, assignee_time_series = calculate_centrality(network, cpc_time_series, assignee_time_series)


