def find_intersection(set1, set2):
    return len([num for num in set1 if num in set2])

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
    ''' Find assignees tied to the technology clusters in which we are interested.
    :return dictionary of assignees with their emergingness level and shared patents for clusterassignees with their emergingness level and shared patents for clusters
    '''

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
    '''
    :return list of cpc subgroups
    '''
    cpc_subgroups = list()

    for patent in topical_patents:
        try:
            cpc_subgroups.append(tensor_patent_cpc_sub[patent])
        except:
            pass

    return list(set(cpc_subgroups))


