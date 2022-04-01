from functions.config import *


def find_intersection(set1, set2):
    """Find intersection of two lists"""

    return len([num for num in set1 if num in set2])


def get_cpc_nodes(topical_clusters, cpc_time_series):
    """Creates CPC nodes for graph"""
    
    cpc_nodes = list()

    for cluster, value in topical_clusters.items:

        year = job_config.data_upload_date.year
        cpc_info = (cluster, {"emergingness": cpc_time_series[cluster][year]["emergingness"],
                              "patent_count": cpc_time_series[cluster][year]["patent_count"],
                              "influence": value})
        cpc_nodes.append(cpc_info)
    
    return cpc_nodes


def get_assignee_nodes(topical_assignees):
    """Creates assignee nodes for graph"""

    assignee_nodes = list()
    for assignee in topical_assignees.keys():
        assignee_value_list = topical_assignees[assignee]["emergingness"]
        assignee_info = (assignee, {"weight": sum(assignee_value_list)/len(assignee_value_list)})
        assignee_nodes.append(assignee_info)

    return assignee_nodes


def get_edge_data(topical_assignees):
    """Creates edges for graph"""

    edges = list()

    for assignee in topical_assignees:
        
        clusters = list(topical_assignees[assignee].keys())
        clusters.remove("emergingness")
        clusters.remove("patents")

        for cluster in clusters:
            edge_info = (assignee, cluster, topical_assignees[assignee][cluster])
            edges.append(edge_info)

    return edges


def get_assignee_data(cluster, patent, patent_value, assignee, topical_assignees):
    """
    Calculates the emergingness level per assignee
    :return: topical assignees dictionary with respective mean emergingness level
                {assignee: {cluster: XX, cluster: YY, emergingness: []}}
    """

    if assignee in topical_assignees:

        # cluster connections
        if cluster in topical_assignees[assignee]:
            topical_assignees[assignee][cluster] += 1
        else:
            topical_assignees[assignee][cluster] = 1

        # assignee emergingness value
        if patent not in topical_assignees[assignee]["patents"]:

            topical_assignees[assignee]["emergingness"].append(patent_value)
            topical_assignees[assignee]["patents"].append(patent)

    else:

        topical_assignees[assignee] = {"emergingness": [patent_value], "patents": [patent], cluster: 1}

    return topical_assignees


def find_topical_assignees(topical_clusters, cpc_time_series, tensor_patent_assignee, tensor_patent):
    """
    Find assignees tied to the topical clusters of the job
    :return: dictionary of topical assignees with their emergingness level and shared patents with each cluster
    """

    topical_assignees = dict()

    for cluster in topical_clusters.keys():
        print("6.2.1 Finding topical assignees for cluster {} ({})".format(cluster, datetime.now()))

        for patent in cpc_time_series[cluster]["patents_final_year"]:

            try:
                assignees = tensor_patent_assignee[patent]
            except:
                print(f'{patent}: no assignee')
                continue

            try: 
                patent_value = tensor_patent[patent]["output"]
            except Exception as e:
                print(f'{patent}:{e}') 
                patent_value = None

            for assignee in assignees:
                topical_assignees = get_assignee_data(cluster, patent, patent_value, assignee, topical_assignees)

    return topical_assignees


def find_topical_clusters(topical_patents, tensor_patent_cpc_sub):
    """
    Finds all CPC subgroups related to the topical patents. Topical patents link to CPC subgroups, and the latter are
    weighted as a sum of the topical patent values
    :return: dictionary of topical clusters and their assigned weight
    """

    cpc_subgroups = dict()

    for patent, value in topical_patents.items():

        try:
            patent_cpc_subgroups = tensor_patent_cpc_sub[patent]

            for group in patent_cpc_subgroups:

                if group not in cpc_subgroups.keys():
                    cpc_subgroups[group] = value
                else:
                    cpc_subgroups[group] += value

        except:
            pass

    return cpc_subgroups
