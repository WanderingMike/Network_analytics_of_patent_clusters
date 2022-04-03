from functions.functions_network_analysis import *


def technology_index(topical_clusters, cpc_time_series, tensors_cpc_sub_patent):
    """
    Creates dataframe with all clusters tied to topical patents. For each cluster, this function retrieves:
    1) The total patent count
    2) The emergingness value of its patents in the last year
    3) The growth in emergingness between the last 2 years
    4) The Technology Index: mean 3-year emergingness growth * mean 3-year patent count growth
    """

    cluster_descriptions = pd.read_csv("data/patentsview_data/cpc_subgroup.tsv",
                                       sep='\t',
                                       header=0,
                                       names=['CPC', 'desc'])

    clusters_df = pd.DataFrame(columns=['subgroup', 'count', 'emergingness', 'delta', 'tech index', 'data'])
    
    # CPC
    clusters_df['subgroup'] = topical_clusters.keys()
    print("length of topical_clusters is {}".format(len(topical_clusters.keys())))
    # desc
    clusters_df = pd.merge(clusters_df, cluster_descriptions, how='left', left_on='subgroup', right_on='subgroup')
    print(clusters_df)
    # count
    clusters_df['count'] = clusters_df['subgroup'].apply(lambda x: len(tensors_cpc_sub_patent[x]))
    # emergingness
    end_year = job_config.data_upload_date.year
    clusters_df['emergingness'] = clusters_df['subgroup'].apply(lambda x: cpc_time_series[x][end_year]['emergingness'])
    # delta
    clusters_df['delta'] = clusters_df['subgroup'].apply(lambda x: cpc_time_series[x][end_year]['emergingness'] -
                                                         cpc_time_series[x][end_year-1]['emergingness'])

    def calculate_technology_index(cpc_subgroup):
        print_value = False
        if cpc_subgroup in ["G06F16/9035", "H04W12/55", "Y10S707/915"]:
            print_value = True
        padding = cpc_time_series[cpc_subgroup]
        show_value(print_value, padding)
        value = list()
        data_aggregate = list()

        def check_validity(pad1, pad2):
            try:
                test = [pad1["emergingness"], pad2["emergingness"], pad1["patent_count"], pad2["patent_count"]]
            except:
                return None, None
            return test

        for i in range(3):
            year = end_year-i
            n = padding[year]
            n_1 = padding[year-1]

            dummy = check_validity(n, n_1)
            if dummy:

                current_em = n["emergingness"]
                prev_em = n_1["emergingness"]
                if prev_em == 0:
                    prev_em = 0.05

                growth_em = current_em / prev_em
                growth_em_penalised = growth_em / (1 + math.exp(5-10*prev_em))
                
                current_count = n["patent_count"]
                prev_count = n_1["patent_count"]
                if prev_count == 0:
                    prev_count = 1

                growth_count_penalised = (0.1*current_count*prev_count) / (prev_count*(0.1 + prev_count))
                
                data_aggregate.append(dummy)
                value.append(growth_em_penalised * growth_count_penalised)
        show_value(print_value, [data_aggregate, value])
        if len(value) >= 1:
            return sum(value)/len(value), data_aggregate
        else:
            return None, None

    # technology index
    clusters_df['tech index'], clusters_df["data"] = zip(*clusters_df['CPC'].apply(calculate_technology_index))

    return clusters_df


def assignee_index(topical_assignees, tensor_assignee):
    """
    Creates a dataframe with all assignees tied to topical patents. For each assignee, this function retrieves:
    1) Name of company
    2) Patent value of assignee patents in latest year
    """

    assignees_df = pd.DataFrame(columns=['ID', 'name', 'value', 'count', 'impact', 'normalised impact', 'influence'])

    # ID
    assignees_df['ID'] = topical_assignees.keys()
    # name
    assignees_df['name'] = assignees_df['ID'].apply(lambda x: tensor_assignee[x])
    # count
    assignees_df['count'] = assignees_df['ID'].apply(lambda x: len(topical_assignees[x]['patent_value']))

    # emergingness
    def assignee_emergingness(row):
        if row["count"] > 10 and row["count"] < 13:
            print(topical_assignees[row["ID"]]["emergingness"], row["count"])
        return sum(topical_assignees[row['ID']]['emergingness']) / row['count']

    assignees_df['emergingness'] = assignees_df.apply(assignee_emergingness, axis=1)

    return assignees_df


def impact_index(node, network, print_value):
    """Calculates the impact value used in the impact and normalised impact indices.
    :return: impact value, number of shared patents to find normalised impact"""
    show_value(print_value, node)
    name = node[0]
    assignee_value = node[1]['weight']
    impact = 0
    count = 0
    show_value(print_value, network[name].items())
    
    for neighbour, value in network[name].items():
        edge_weight = value['weight']
        node_emergingness = network.nodes[neighbour]["emergingness"]
        node_influence = network.nodes[neighbour]["influence"]
        count += edge_weight
        impact += assignee_value * edge_weight * node_emergingness * node_influence
    if len(network[name].keys()) > 5 and len(network[name].keys()) < 11:
        print(network[name].items)
        print(impact, count)

    return impact, count


def network_indices(cpc_nodes, assignee_nodes, edges, assignee_df):
    """
    Retrieves remaining assignee indices:
    3) Impact: Assignee emergingness level * CPC cluster emergingness * number of shared patents
    4) Normalised Impact: Impact / number of patents shared
    5) Influence: Katz Centrality measure
    """

    # Creating Graph
    print(f'6.7.1 Creating graph ({datetime.now()})')
    network = nx.Graph()
    network.add_nodes_from(cpc_nodes)
    network.add_nodes_from(assignee_nodes)
    network.add_weighted_edges_from(edges)
    save_pickle("data/ultimate/{}.pkl".format(job_config.graph_name), network)

    # impact
    for node in assignee_nodes:
        if node == assignees_nodes[0]:
            impact, length = impact_index(node, network, True)
        else:
            impact, length = impact_index(node, network, False)
        assignee_df.loc[assignee_df['ID'] == node[0], 'impact'] = impact
        assignee_df.loc[assignee_df['ID'] == node[0], 'normalised impact'] = impact / length

    # Katz centrality
    centrality = nx.eigenvector_centrality(network, weight="weight", max_iter=10000)

    for node, centrality_measure in sorted(centrality.items()):
        assignee_df.loc[assignee_df['ID'] == node, 'influence'] = centrality_measure

    return assignee_df


def unfold_network(cpc_time_series, full_tensors, topical_patents):
    """
    This function has four steps:
     1) Retrieving/calculating relevant data for all assignees, cpc patent clusters and patents themselves.
     2) Setting up network/graph
     3) Centrality analytics
     4) Further analytics and plots
    """

    # Building Network
    print("6.1 Finding topical clusters ({})".format(datetime.now())) 
    topical_clusters = find_topical_clusters(topical_patents, full_tensors["patent_cpc_sub"])

    print("6.2 Finding topical assignees ({})".format(datetime.now())) 
    topical_assignees = find_topical_assignees(topical_clusters,
                                               cpc_time_series,
                                               full_tensors["patent_assignee"],
                                               full_tensors["patent"])

    print("6.4 Getting nodes and edges ({})".format(datetime.now()))
    cpc_nodes = get_cpc_nodes(topical_clusters, cpc_time_series)
    assignee_nodes = get_assignee_nodes(topical_assignees)
    edges = get_edge_data(topical_assignees)
   
    # Indices
    print(f'6.5 Calculating Technology Index ({datetime.now()})')
    clusters_df = technology_index(topical_clusters, cpc_time_series, full_tensors["cpc_sub_patent"])

    print(f'6.6 Calculating Assignee Index ({datetime.now()})')
    assignee_df = assignee_index(topical_assignees, full_tensors["assignee"])

    print(f'6.7 Calculating Impact Index ({datetime.now()})')
    assignee_df = network_indices(cpc_nodes, assignee_nodes, edges, assignee_df)
    
    # Output
    print(f'6.10 Writing to output files ({datetime.now()})')
    print(clusters_df)
    clusters_df.to_csv("output_tables/clusters_df.csv")
    print(assignee_df)
    assignee_df.to_csv("output_tables/assignee_df.csv")

    clusters_df.sort_values("tech index", inplace=True, ascending=False)
    clusters_df.to_markdown("output_tables/technologies_index.md")

    assignee_df.sort_values("emergingness", inplace=True, ascending=False)
    assignee_df.to_markdown("output_tables/assignee_index.md")

    assignee_df.sort_values("impact", inplace=True, ascending=False)
    assignee_df.to_markdown("output_tables/impact_index.md")

    assignee_df.sort_values("normalised impact", inplace=True, ascending=False)
    assignee_df.to_markdown("output_tables/norm_impact_index.md")

    assignee_df.sort_values("influence", inplace=True, ascending=False)
    assignee_df.to_markdown("output_tables/influence_index.md")
