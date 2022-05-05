from functions.functions_main import *


def managerial_layer():
    """
    The main function of our system. Step-by-step walk through:
    1) Loading all tensors created by tensor_deployment.py
    2) Execute the classification algorithm with run_ml()
    3) For each job, find patents related to the keywords
    4) For those patents, map the network of relevant technologies and assignees
    """

    if job_config.load_main:

        cpc_time_series = load_pickle("data/clusters.pkl")
        tensors = load_pickle("data/tensors.pkl")

    else:

        print("1. Loading tensors ({})".format(datetime.now()))

        tensors = {
            "assignee": None,
            "cpc_sub_patent": None,
            "patent_cpc_main": None,
            "patent_cpc_sub": None,
            "otherreference": None,
            "patent": None,
            "patent_assignee": None,
            "assignee_patent": None,
            "inventor": None,
            "forward_citation": None,
            "backward_citation": None
        }
     
        for key in tensors.keys():
            tensors[key] = load_pickle("data/tensors/{}.pkl".format(key))

        print("2. Preparing CPC clusters ({})".format(datetime.now()))

        cpc_time_series, tensors["patent"] = run_ml(tensors)

        save_pickle("data/clusters.pkl", cpc_time_series)
        save_pickle("data/tensors.pkl", tensors)

    print("3. Finished preparing data ({})".format(datetime.now()))

    for keywords in job_config.keyphrases:
        
        print("4. Running job {} ({})".format(keywords, datetime.now()))

        print("5. Finding topical patents ({})".format(datetime.now()))
        if job_config.load_topical_patents:

            topical_patents = load_pickle("data/topical_patents.pkl")

        else:

            topical_patents = finding_topical_patents(tensors["patent"], keywords)
            save_pickle("data/topical_patents.pkl", topical_patents)

        print("6. Unfolding network ({})".format(datetime.now()))
        unfold_network(cpc_time_series, tensors, topical_patents)


def inspect_network(name):
    """Prints most important technologies per assignee or most important companies per technology."""

    graph = load_pickle("data/{}".format(name))
    cluster_descriptions = pd.read_csv("data/patentsview_data/cpc_subgroup.tsv", sep='\t', header=0,
                                       names=['id', 'desc'])
    assignee_descriptions = pd.read_csv("data/patentsview_data/assignee.tsv", sep='\t', header=0,
                                        usecols=[0, 4],
                                        names=['id', 'desc'])

    loop = 'y'
    while loop == 'y':

        # Print ranking?
        selection = input("Give company or cluster ID\n")
        try:
            if len(selection) == 36:
                ranking = publish_ranking(graph, selection, cluster_descriptions)
            else:
                ranking = publish_ranking(graph, selection, assignee_descriptions)
        except:
            continue

        # Save ranking?
        save_ranking(ranking)

        # Continue?
        loop = input("Would you like to continue? (y or n)  ")


def display_topical_abstracts():
    """This function is useful when trying to understand the technologies index and curating the keywords list.
    Given a set of CPC subgroups, this function prints all topical patents that were linked to these subgroups."""

    topical_patents = load_pickle("data/topical_patents.pkl")
    print("Loading required tensors")
    tensor_patent_cpc_sub = load_pickle("data/tensors/patent_cpc_sub.pkl")
    tensor_patent = load_pickle("data/tensors/patent.pkl")

    loop = 'y'
    while loop == 'y':
        subgroup_set = input("Set of comma-separated subgroups: ").split(",")
        for patent, reference_count in topical_patents.items():
            if reference_count > 0:
                try:
                    if find_intersection(tensor_patent_cpc_sub[patent], subgroup_set) > 0:
                        print("#"*30)
                        print(patent)
                        print(tensor_patent[patent]["abstract"])
                except:
                    continue

        # Continue?
        loop = input("Would you like to continue? (y or n)  ")


def curate_ranking():
    """Develop new rankings"""

    loop = 'y'
    while loop == 'y':

        df_name = input("Which dataframe? \n (1) Technologies\n (2) Companies\n")
        if df_name == "1":
            df = pd.read_csv("output_tables/clusters_df.csv", index_col=0, header=0)
        else:
            df = pd.read_csv("output_tables/assignee_df.csv", index_col=0, header=0)
        print(df)

        filter_var = input("Filter on variable: ")
        filter_order = input("(1) Ascending, (2) Descending: ")
        min_val = input("Variable - Minimum value (separated by white space): ").split()

        df = df[df[min_val[0]] > float(min_val[1])]
        df.sort_values(filter_var, inplace=True, ascending=True if filter_order == 1 else False)
        print(df)
        save_ranking(df)

        # Continue?
        loop = input("Would you like to continue? (y or n)  ")


def interaction():
    """Displays menu and runs appropriate functions according to user wishes"""

    ans = input("Would you like to: \n"
                "(1) Create a new graph\n" +
                "(2) Inspect an assignee or a technology\n" +
                "(3) Find abstract of topical patents pertaining to a group of technologies\n" +
                "(4) Publish a new index ranking\n")

    if ans == '1':
        managerial_layer()
    elif ans == '2':
        network_name = input("Network file name: ")
        inspect_network(network_name)
    elif ans == '3':
        display_topical_abstracts()
    elif ans == '4':
        curate_ranking()
    else:
        print("Invalid input")


if __name__ == "__main__":
    display_topical_abstracts()
