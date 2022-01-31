from ML import *
from functions.functions_network_analysis import *
import networkx as nx


def prepare_assignee_series(tensor_assignee_patent, tensor_year_patent):
    patents_for_last_year = tensor_year_patent[2021]
    for patent in patents_for_last_year:
        if not patent.isnumeric():
            print(patent)


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


def unfold_network(start, end, loading=False):
    ''' This function has four steps:
     1) Retrieving/calculating relevant data for all assignees, cpc patent clusters and patents themselves.
     2) Setting up network/graph
     3) Centrality analytics
     4) Further analytics and plots
    '''
    
    # Loading tensors
    tensor_list = ["assignee", "assignee_patent", "cpc_patent", "patent", "year_patent"]
    tensors = {k: None for k in tensor_list}
    for tensor in tensor_list:
        tensors[tensor] = load_tensor(tensor)
    prepare_assignee_series(tensors["assignee_patent"], tensors["year_patent"])

    # Fetching CPC
    print("Preparing CPC clusters")
    print(datetime.now())
    
    if loading:
        ffile = open("data/clusters_3.pkl", "rb")
        cpc_time_series = pickle.load(ffile)

    else:
        cpc_time_series = prepare_time_series(start, end)
        a_file = open("data/clusters.pkl", "wb")
        pickle.dump(cpc_time_series, a_file)
        a_file.close()
        #print_output("std_out/process/ml_main")

    print("Finished CPC clusters")
    print(datetime.now())

    # Fetching assignees
    assignee_series = dict()
    for assignee, patents in tensors["assignee_patent"].items():
        if len(patents) > 200:
            assignee_series[assignee] = patents
    print("Finished assignee clusters")
    print(datetime.now())

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


if __name__ == "__main__":
    start = datetime(1970,1,1)
    end = datetime(2021,12,31)
    unfold_network(start, end, loading=True) # $ make sure start and end date make sense $
