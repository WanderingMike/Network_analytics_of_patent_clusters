from functions.config import *


def find_intersection(set1, set2):
    """Find intersection of two lists"""

    return len([num for num in set1 if num in set2])


def get_cpc_nodes(topical_clusters, cpc_time_series):
    """Creates CPC nodes for graph"""
    
    cpc_nodes = list()

    for cluster, value in topical_clusters.items():
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

    for assignee in topical_assignees.keys():
        
        clusters = list(topical_assignees[assignee].keys())
        clusters.remove("emergingness")
        clusters.remove("patents")

        for cluster in clusters:
            edge_info = (assignee, cluster, topical_assignees[assignee][cluster])
            edges.append(edge_info)

    return edges


def calculate_technology_index(cpc_subgroup, padding):
    """Calculates the technology index for a specific cpc_subgroup
    :param cpc_subgroup: subgroup ID
    :param padding: dataset with only the required timeseries of the specific cpc subgroup
    """
    
    value = list()
    data_aggregate = list()

    end_year = job_config.data_upload_date.year

    for i in range(3, 0, -1):
        year = end_year-i
        n = padding[year+1]
        n_1 = padding[year]

        try:
            
            current_count = n["patent_count"]
            prev_count = n_1["patent_count"]
            
            if current_count < 4 or prev_count < 4:
                continue
            
            leverage = prev_count**(1/5)
            growth_count_penalised = (current_count/prev_count) * leverage

            current_em = n["emergingness"]
            prev_em = n_1["emergingness"]
            if prev_em == 0:
                prev_em = 0.1

            growth_em = current_em / prev_em

            data_aggregate.append([n_1["emergingness"], n["emergingness"], n_1["patent_count"], n["patent_count"]])
            value.append(growth_em * growth_count_penalised)
            print("#"*30)
            print(data_aggregate)
            print(value)

        except Exception as e:
            print(cpc_subgroup, e)

    if len(value) >= 1:
        return sum(value)/len(value)
    else:
        return None


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
    :return: dictionary of topical assignees with their emergingness level and shared patents with each cluster. This data only considers links to topical clusters with the most recent patents.
    """

    topical_assignees = dict()
    total_patents = 0
    patents_no_assignee = 0

    print("6.2.1 Finding topical assignees for cluster ({})".format(datetime.now()))

    for cluster in topical_clusters.keys():

        for patent in cpc_time_series[cluster]["patents_final_year"]:
            try:
                assignees = tensor_patent_assignee[patent]
            except:
                patents_no_assignee += 1
                continue

            try: 
                patent_value = tensor_patent[patent]["output"]
            except Exception as e:
                patent_value = None

            for assignee in assignees:
                topical_assignees = get_assignee_data(cluster, patent, patent_value, assignee, topical_assignees)
            total_patents += 1

    print("Patents missing an assignee: {}/{}".format(patents_no_assignee, total_patents))
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

            if "nan" in patent_cpc_subgroups:
                continue

            for group in patent_cpc_subgroups:

                if group not in cpc_subgroups.keys():
                    cpc_subgroups[group] = value
                else:
                    print("Adding to group {} - {} - +{}".format(group, cpc_subgroups[group], value))
                    cpc_subgroups[group] += value

        except Exception as e:
            print(e)

    return cpc_subgroups
