import string
from network_analysis import *
from ML import *

def load_tensor(tensor_key):
    '''
    This function loads a data tensor produced by the tensor_deployment.py file.
    :param tensor_key: name of tensor
    '''
    file = open("data/patentsview_cleaned/{}.pkl".format(tensor_key), "rb")
    tensor = pickle.load(file)

    return tensor


def search_abstract(patent, value, concepts):
    '''
    Takes a list of words or ideas and looks for them in the patent abstract.
    :param patent: patent dictionary key 
    :param value: patent dictionary value
    :param concepts: words to find in patent abstract
    '''

    abstract = value["abstract"]
    abstract_cleaned = abstract.translate(str.maketrans('','', string.punctuation))
    abstract_lower = abstract_cleaned.lower()
    abstract_tokenised = abstract_lower.split(' ')
    
    for concept in concepts:
        concept_sep = concept.split(' ')
        first_word = concept_sep[0]
        for i in range(len(abstract_tokenised)):
            if abstract_tokenised[i] == first_word:
                extracted_token = abstract_tokenised[i:i+len(concept_sep)]
                if ' '.join(extracted_token) == concept:
                    return True

    return False


def finding_topical_patents(tensor_patent, keywords):
    '''
    Finds all patents that contain at least of one the keyworks.
    :param tensor_patent: tensor which contains patent abstracts
    :param keywords: list of words to search for
    '''
    concepts = [words.split(' ') for words in keywords]

    topical_patents = list()
    
    for patent, value in tqdm(list(tensor_patent.items())[:5]):

        try:
            if(search_abstract(patent, value, concepts)):
                topical_patents.append(patent)
        except:
            pass

    return list(set(topical_patents))


def managerial_layer(loading=False):
    '''The main function of our system. Step-by-step walkthrough:
    1) Loading all tensors created by tensor_deployment.py
    2) Execute the classification algorithm with run_ML()
    3) For each job, find patents related to the keywords
    4) For those patents, map the network of relevant technologies and assignees
    ''' 

    if loading:

        ffile = open("data/clusters.pkl", "rb")
        cpc_time_series = pickle.load(ffile)

        ffile2 = open("data/tensors.pkl", "rb")
        tensors = pickle.load(ffile2)
        for i in [10, 10000, 1000000, 5000000]:
            res = list(tensors["patent"].keys())[i]
            print(tensors["patent"][res])
        res2 = list(cpc_time_series.keys())[10000]
        print(cpc_time_series[res2])

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
            tensors[key] = load_tensor(key)

        print("2. Preparing CPC clusters ({})".format(datetime.now()))

        cpc_time_series, tensors = run_ML(tensors, job_config.start, job_config.end)
        a_file = open("data/clusters.pkl", "wb")
        pickle.dump(cpc_time_series, a_file)
        a_file.close()
        
        b_file = open("data/tensors.pkl", "wb")
        pickle.dump(tensors, b_file)
        b_file.close()

    print("3. Finished preparing data ({})".format(datetime.now()))

    for keywords in job_config.jobs:
        
        print("4. Running job {} ({})".format(keywords, datetime.now()))

        print("5. Finding topical patents ({})".format(datetime.now()))
        topical_patents = finding_topical_patents(tensors["patent"], keywords)
        print(topical_patents)
        print("6. Unfolding network ({})".format(datetime.now()))
        unfold_network(cpc_time_series, tensors, topical_patents, job_config.end.year)


if __name__ == "__main__":
    managerial_layer(loading=False)
