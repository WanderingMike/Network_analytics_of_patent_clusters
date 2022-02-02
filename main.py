import string
from network_analysis import *

def load_tensor(tensor_key):

    file = open("data/patentsview_cleaned/{}.pkl".format(tensor_key), "rb")
    tensor = pickle.load(file)

    return tensor


def search_abstract(patent, value):

    abstract = value["abstract"]
    abstract_cleaned = abstract.translate(str.maketrans('','', string.punctuation))
    abstract_lower = abstract_cleaned.lower()
    abstract_tokenised = abstract_lower.split(' ')
    
    for concept in concepts:
        first_word = concept[0]
        for i in range(len(abstract_tokenised)):
            if abstract_tokenised[i] == first_word:
                extracted_token = abstract_tokenised[i:i+len(concept)]
                if ' '.join(extracted_token) == ' '.join(concept):
                    return patent

    return None


def finding_topical_patents(tensor_patent, keywords):
    
    concepts = [words.split(' ') for words in keywords]

    topical_patents = list()
    
    for patent, value in tensor_patent.items():
        patent = topic_found = search_abstract(patent, value)
        if patent:
            topical_patents.append(patent)

    return list(set(topical_patents))


def managerial_layer(start, end, jobs, loading=False):

    print("1. Loading tensors")

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

    # Fetching CPC
    print("2. Preparing CPC clusters ({})".format(datetime.now()))
    
    if loading:

        ffile = open("data/clusters.pkl", "rb")
        cpc_time_series = pickle.load(ffile)

    else:

        cpc_time_series, tensors = run_ML(tensors, start, end)
        a_file = open("data/clusters.pkl", "wb")
        pickle.dump(cpc_time_series, a_file)
        a_file.close()

    print("3. Finished CPC clusters ({})".format(datetime.now()))

    for keywords in jobs:
        
        print("4. Running job {} ({})".format(keywords, datetime.now()))

        print("5. Finding topical patents ({})".format(datetime.now()))
        topical_patents = finding_topical_patents(tensors["patent"], keywords)

        print("6. Unfolding network ({})".format(datetime.now()))
        unfold_network(cpc_times_series, tensors, topical_patents)


if __name__ == "__main__":
    start = datetime(2021,1,1)
    end = datetime(2021,12,31)
    jobs = [["cyber", "honeypot", "cybersecurity"],
            ["quantum", "computer", "quantum mechanics"],
            ["5G"]]
    managerial_layer(start, end, jobs=jobs)
