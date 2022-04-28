from functions.config import *


def find_intersection(set1, set2):
    """Find intersection of two lists"""

    return len([num for num in set1 if num in set2])


def get_cpc_nodes(topical_clusters, cpc_time_series):
    """Creates CPC nodes for graph"""
    
    cpc_nodes = list()

    for cluster, value in topical_clusters.items():
        print_value = False
        if cluster == list(topical_clusters.keys())[0]:
            print_value = True
        show_value(print_value, cpc_time_series[cluster]) 
        year = job_config.data_upload_date.year
        cpc_info = (cluster, {"emergingness": cpc_time_series[cluster][year]["emergingness"],
                              "patent_count": cpc_time_series[cluster][year]["patent_count"],
                              "influence": value})
        cpc_nodes.append(cpc_info)
        show_value(print_value, cpc_nodes)
    
    return cpc_nodes


def get_assignee_nodes(topical_assignees):
    """Creates assignee nodes for graph"""

    assignee_nodes = list()
    for assignee in topical_assignees.keys():
        print_value = False
        if assignee == list(topical_assignees.keys())[0]:
            print_value = True
        show_value(print_value, topical_assignees[assignee]) 
        assignee_value_list = topical_assignees[assignee]["emergingness"]
        assignee_info = (assignee, {"weight": sum(assignee_value_list)/len(assignee_value_list)})
        assignee_nodes.append(assignee_info)
        show_value(print_value, topical_assignees[assignee])

    return assignee_nodes


def get_edge_data(topical_assignees):
    """Creates edges for graph"""

    edges = list()

    for assignee in topical_assignees.keys():
        
        print_value = False
        if assignee == list(topical_assignees.keys())[0]:
            print_value = True
        show_value(print_value, topical_assignees[assignee].keys()) 
        clusters = list(topical_assignees[assignee].keys())
        clusters.remove("emergingness")
        clusters.remove("patents")

        for cluster in clusters:
            edge_info = (assignee, cluster, topical_assignees[assignee][cluster])
            edges.append(edge_info)
        show_value(print_value, edges)

    return edges


def calculate_technology_index(cpc_subgroup, padding):
    '''comment'''
    
    print_value = False
    if cpc_subgroup in ["G06F16/9035", "H04W12/55", "Y10S707/915"]:
        print_value = True
    show_value(print_value, padding)
    value = list()
    data_aggregate = list()

    def check_validity(pad1, pad2):
        try:
            test = [pad1["emergingness"], pad2["emergingness"], pad1["patent_count"], pad2["patent_count"]]
            return test
        except:
            return None

    end_year = job_config.data_upload_date.year
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
    total_patents = 0
    patents_no_assignee = 0

    for cluster in topical_clusters.keys():
        print("6.2.1 Finding topical assignees for cluster {} ({})".format(cluster, datetime.now()))

        print_value = False
        if cluster == "B60L2270/32":
            print_value = True
            show_value(print_value, cpc_time_series[cluster]["patents_final_year"])

        for patent in cpc_time_series[cluster]["patents_final_year"]:
            try:
                assignees = tensor_patent_assignee[patent]
            except:
                print(f'{patent}: no assignee')
                patents_no_assignee += 1
                continue

            try: 
                patent_value = tensor_patent[patent]["output"]
            except Exception as e:
                print(f'{patent}:{e}') 
                patent_value = None

            for assignee in assignees:
                topical_assignees = get_assignee_data(cluster, patent, patent_value, assignee, topical_assignees)
            total_patents += 1
            show_value(print_value, topical_assignees)

    print("Results: {}/{}".format(patents_no_assignee, total_patents))
    return topical_assignees


def find_topical_clusters(topical_patents, tensor_patent_cpc_sub):
    """
    Finds all CPC subgroups related to the topical patents. Topical patents link to CPC subgroups, and the latter are
    weighted as a sum of the topical patent values
    :return: dictionary of topical clusters and their assigned weight
    """
    print("Topical Clusters")
    cpc_subgroups = dict()
    for patent, value in topical_patents.items():
        print_value = False
        if patent == list(topical_patents.keys())[0]:
            print_value = True
        try:
            patent_cpc_subgroups = tensor_patent_cpc_sub[patent]
            if "nan" in patent_cpc_subgroups:
                continue
            show_value(print_value, patent_cpc_subgroups)
            for group in patent_cpc_subgroups:
                if group not in cpc_subgroups.keys():
                    cpc_subgroups[group] = value
                else:
                    cpc_subgroups[group] += value
            show_value(print_value, cpc_subgroups)

        except Exception as e:
            print(e)

    return cpc_subgroups
