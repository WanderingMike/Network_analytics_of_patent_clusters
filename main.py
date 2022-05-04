from network_analysis import *
from ML import *


def search_abstract(abstract, concepts):
    """
    Takes a list of words or ideas and looks for them in the patent abstract. Use Term Frequency to normalise.
    :param value: patent dictionary value
    :param concepts: words to find in patent abstract
    """

    abstract_cleaned = abstract.translate(str.maketrans('', '', string.punctuation))
    abstract_lower = abstract_cleaned.lower()
    abstract_tokenised = abstract_lower.split(' ')
    references_count = 0
    
    for concept in concepts:
        first_word = concept[0]
        for word_loc in range(len(abstract_tokenised)):
            if abstract_tokenised[word_loc] == first_word:
                extracted_token = abstract_tokenised[word_loc:word_loc+len(concept)]
                if ' '.join(extracted_token) == ' '.join(concept):
                    references_count += 1

    return 100*references_count/len(abstract_tokenised)  # abstract is ~400 words, 1-2 cyberwords: ~1/100


def check_category(categories):
    lst_cat = ['B60R25/104',
             'B01J2219/00704',
             'B60R25/2072',
             'A63G31/16',
             'B60R25/406',
             'A63B21/0455',
             'A63B2208/0209',
             'C12N2310/321',
             'C12N15/70',
             'C12N2310/16',
             'A47J37/1266',
             'B41J29/38',
             'B32B7/12',
             'A23L33/135',
             'H01F2005/027',
             'A47G2029/145',
             'A44C5/0007',
             'B32B2307/42',
             'B32B2307/4023',
             'Y02A40/22',
             'E05B73/0005',
             'E05F15/684',
             'A61B1/0669',
             'B32B29/02',
             'G07D7/0032',
             'E05B45/005',
             'G01F1/007',
             'G01R22/06',
            'B32B9/045']
    return [x for x in categories if x in lst_cat]


def finding_topical_patents(tensor_patent, tensor_patent_cpc_sub, keywords):
    """
    Finds all patents that contain at least of one the keywords.
    :param tensor_patent: tensor which contains patent abstracts
    :param keywords: list of words to search for
    """
    concepts = [words.split(' ') for words in keywords]

    topical_patents = dict()
    
    for patent, value in tensor_patent.items():
        
        try:
            reference_count = search_abstract(value["abstract"], concepts)
            answer = check_category(tensor_patent_cpc_sub[patent])
            if reference_count != 0 and len(answer) > 0:
                print("#"*30)
                print(answer)
                print(value["abstract"])
                topical_patents[patent] = reference_count
        except:
            continue

    return topical_patents


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

            topical_patents = finding_topical_patents(tensors["patent"], tensors["patent_cpc_sub"], keywords)
            save_pickle("data/topical_patents.pkl", topical_patents)
        input("hey") 
        print("6. Unfolding network ({})".format(datetime.now()))
        unfold_network(cpc_time_series, tensors, topical_patents)


def publish_ranking(graph, node_name, node_descriptions):
    """
    Creates ranked dataset of neighbouring graph nodes
    :param graph: query graph
    :param node_name: assignee or CPC ID
    :param node_descriptions: dataset with all adequate node descriptions
    :return: ranking as a pandas dataframe
    """

    ranking = pd.DataFrame.from_dict(graph[node_name], orient="index")

    # header
    print("#"*50)
    print(node_name)

    # ranking
    ranking = pd.merge(ranking, node_descriptions, how='left', left_index=True, right_on='id')
    ranking.sort_values(by=["weight"], inplace=True, ascending=False)
    print(ranking)

    return ranking


def inspect_network():
    """Prints most important technologies per assignee or most important companies per technology."""

    graph = load_pickle("data/network.pkl")
    cluster_descriptions = pd.read_csv("data/patentsview_data/cpc_subgroup.tsv", sep='\t', header=0,
                                       names=['id', 'desc'])
    assignee_descriptions = pd.read_csv("data/patentsview_data/assignee.tsv", sep='\t', header=0,
                                        usecols=[0, 4],
                                        names=['id', 'desc'])

    loop = 'y'
    while loop == 'y':

        # Print ranking?
        selection = input("Give company or cluster ID\n")
        if len(selection) == 36:
            ranking = publish_ranking(graph, selection, cluster_descriptions)
        else:
            ranking = publish_ranking(graph, selection, assignee_descriptions)

        # Save ranking?
        save = input("Would you like to save your ranking? Give address if yes.\n")
        if save:
            try:
                ranking.to_csv("{}".format(save))
            except:
                print("Folder address not found.\n")

        # Continue?
        loop = input("Would you like to continue? (y or n)  ")


if __name__ == "__main__":
    managerial_layer()
    #inspect_network()
    # if (answer:=input("Would you like to create (1) or load (2)")) == 1:
    #     managerial_layer()
    # elif answer == 2:
    #     #network_name = input("")
    #     inspect_network()
    # else:
    #     print("Invalid input")
