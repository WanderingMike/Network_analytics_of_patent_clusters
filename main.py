from network_analysis import *
from ML import *


def search_abstract(value, concepts):
    '''
    Takes a list of words or ideas and looks for them in the patent abstract. Use Term Frequency to normalise.
    :param value: patent dictionary value
    :param concepts: words to find in patent abstract
    '''

    abstract = value["abstract"]
    abstract_cleaned = abstract.translate(str.maketrans('','', string.punctuation))
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

    return 100*references_count/len(abstract_tokenised) # abstract is ~400 words, 1-2 cyberwords: ~1/100


def finding_topical_patents(tensor_patent, keywords):
    '''
    Finds all patents that contain at least of one the keyworks.
    :param tensor_patent: tensor which contains patent abstracts
    :param keywords: list of words to search for
    '''
    concepts = [words.split(' ') for words in keywords]

    topical_patents = dict()
    
    for patent, value in tensor_patent.items():
        
        try:
            reference_count = search_abstract(value, concepts)
            if reference_count != 0:
                topical_patents[patent] = search_abstract(value, concepts)
        except:
            continue

    return topical_patents


def managerial_layer():
    '''The main function of our system. Step-by-step walkthrough:
    1) Loading all tensors created by tensor_deployment.py
    2) Execute the classification algorithm with run_ML()
    3) For each job, find patents related to the keywords
    4) For those patents, map the network of relevant technologies and assignees
    ''' 

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

        cpc_time_series, tensors = run_ML(tensors)

        save_pickle("data/clusters.pkl", cpc_time_series)
        save_pickle("data/tensors.pkl", tensors)

    print("3. Finished preparing data ({})".format(datetime.now()))

    for keywords in job_config.keyphrases:
        
        print("4. Running job {} ({})".format(keywords, datetime.now()))

        print("5. Finding topical patents ({})".format(datetime.now()))
        topical_patents = finding_topical_patents(tensors["patent"], keywords)
        print(topical_patents)
        print("6. Unfolding network ({})".format(datetime.now()))
        unfold_network(cpc_time_series, tensors, topical_patents)


def publish_ranking(graph, node_name, node_descriptions):
    '''
    Creates ranked dataset of neighbouring graph nodes
    :param graph: query graph
    :param node_name: assignee or CPC ID
    :param node_descriptions: dataset with all adequate node descriptions
    :return: ranking as a pandas dataframe
    '''

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
    '''Prints most important technologies per assignee or most important companies per technology.'''

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
    inspect_network()
    # if (answer:=input("Would you like to create (1) or load (2)")) == 1:
    #     managerial_layer()
    # elif answer == 2:
    #     #network_name = input("")
    #     inspect_network()
    # else:
    #     print("Invalid input")


