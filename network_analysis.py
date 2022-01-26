from ML import *
from functions.functions_network_analysis import *
import networkx as nx


def prepare_assignee_nodes(start, end, assignee_patent, year_patent):
    '''
    Create a time series of patent count for all assignees that have published at least 50 patents over their lifetime.
    :param start: period start
    :param end: period end
    :param tensors: All four loaded tensors
    :return: time series for all assignees, of the shape {assignee_A: {2000: 10, 2001: 12, 2002: 3, ...}}
    '''
    assignee_nodes = dict()

    years = range(start, end+1)
    for assignee, patents_list in assignee_patent.items():
        if len(patents_list) > 50:
            assignee_nodes[assignee] = {k: None for k in years}
            for year in years:
                common_patents = [patent for patent in patents_list if patent in year_patent]
                assignee_nodes[assignee][year] = len(common_patents)

    return assignee_nodes


def fetch_edge_data(tensors, cpc_time_series, assignee_time_series, start, end):
    '''
    Drawing all network edges for three sets of relationships:
    1) assignee-CPC
    2) CPC-CPC
    3) assignee-assignee
    :param network: network to add edges to
    :param tensors: data to create links
    :return: linked network
    '''

    assignee_list = list(assignee_time_series.keys())
    cpc_list = list(cpc_time_series.keys())
    years = range(start, end+1)
    edges = {k: list() for k in years}

    # assignee-CPC
    for assignee in assignee_list:
        for cpc in cpc_list:
            for year in years:
                weight = find_intersection(tensors["assignee_patent"][assignee],
                                               tensors["cpc_patent"][cpc],
                                               tensors["year_patent"][year])
                edges[year].append((assignee, cpc, weight))


    # assignee-assignee
    for assignee1 in assignee_list:
        assignee_list.remove(assignee1)
        for assignee2 in assignee_list:
            for year in years:
                weight = find_intersection(tensors["assignee_patent"][assignee1],
                                               tensors["assignee_patent"][assignee2],
                                               tensors["year_patent"][year])
                edges[year].append((assignee1, assignee2, weight))


    # CPC-CPC
    for cpc1 in cpc_list:
        cpc_list.remove(cpc1)
        for cpc2 in cpc_list:
            for year in years:
                weight = find_intersection(tensors["cpc_patent"][cpc1],
                                           tensors["cpc_patent"][cpc2],
                                           tensors["year_patent"][year])
                edges[year].append((cpc1, cpc2, weight))

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


def unfold_network(start, end):
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

    # Fetching CPC
    print("Preparing CPC clusters")
    print(datetime.now())
    cpc_time_series = prepare_time_series(start, end)
    print("Finished CPC clusters")
    print(datetime.now())

    # Fetching assignees
    print("Preparing assignee clusters")
    print(datetime.now())
    assignee_time_series = prepare_assignee_nodes(start.year, end.year, tensors["assignee_patent"], tensors["year_patent"])
    print("Finished assignee clusters")
    print(datetime.now())

    # Creating Graph
    network = nx.Graph()
    network.add_nodes_from(cpc_time_series.keys())
    network.add_nodes_from(assignee_time_series.keys())

    print("Fetching edge data")
    print(datetime.now())
    edges = fetch_edge_data(tensors, cpc_time_series, assignee_time_series, start.year, end.year)
    print("Fetched edge data")
    print(datetime.now())

    for year in range(start.year, end.year+1):
        # Draw edges
        network = update_edges(network, edges[year])

        # Centrality analytics
        cpc_time_series, assignee_time_series = calculate_centrality(network, cpc_time_series, assignee_time_series)


if __name__ == "__main__":
    start = datetime(2016,1,1)
    end = datetime(2016,12,31)
    unfold_network(start, end) # $ make sure start and end date make sense $
