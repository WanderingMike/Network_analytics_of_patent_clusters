from datetime import datetime
from functions.config_ML import *

def find_intersection(set1, set2):
    '''Find intersection of two lists'''
    return len([num for num in set1 if num in set2])


def get_cpc_nodes(topical_clusters, cpc_time_series):
    '''Creates CPC nodes for graph'''
    
    cpc_nodes = list()

    for cluster in topical_clusters:

        cpc_info = (cluster, {"emergingness": cpc_time_series[cluster][job_config.upload_date.year]["emergingness"], 
                              "patent_count": cpc_time_series[cluster][job_config.upload_date.year]["patent_count"]})
        cpc_nodes.append(cpc_info)
    
    return cpc_nodes


def get_assignee_nodes(topical_assignees):
    '''Creates assignee nodes for graph'''

    assignee_nodes = list()
    for assignee in topical_assignees.keys():
        assignee_data = topical_assignees[assignee]["emergingness"]
        assignee_info = (assignee, {"weight": sum(assignee_data)/len(assignee_data)})
        assignee_nodes.append(assignee_info)

    return assignee_nodes


def get_edge_data(topical_assignees):
    '''Creates edges for graph'''

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
    '''
    Calculates the emergingness level per assignee
    :return: topical assignees dictionary with respective mean emergingness level {assignee: {cluster: XX, cluster: YY, emergingness: []}}
    '''

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

        topical_assignees[assignee] = {"emergingness": [patent_value], "patents":[patent], cluster: 1}

    return topical_assignees


def find_topical_assignees(topical_clusters, cpc_time_series, tensor_patent_assignee, tensor_patent):
    ''' 
    Find assignees tied to the topical clusters of the job
    :return: dictionary of topical assignees with their emergingness level and shared patents with each cluster
    '''

    topical_assignees = dict()

    for cluster in topical_clusters:
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
    '''
    Finds all CPC subgroups related to the topical patents
    :return: list of topical clusters
    '''
    cpc_subgroups = list()

    for patent in topical_patents:
        try:
            cpc_subgroups += tensor_patent_cpc_sub[patent]
        except:
            pass

    return list(set(cpc_subgroups))



