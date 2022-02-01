import string
from network_analysis import *

def finding_topical_patents(tensor_patent, keywords):
    
    topical_patents = list()

    for patent, value in tensor_patent.items():
        abstract = value["abstract"]
        abstract_cleaned = abstract.translate(str.maketrans('','', string.punctuation))
        abstract_lower = abstract_cleaned.lower()
        abstract_tokenised = abstract_lower.split(' ')
        
        if any(word in abstract_tokenised for word in keywords):

            topical_patents.append(patent)

    return topical_patents


def managerial_layer(start, end, keywords=None, loading=False):

    tensors = {
        "assignee": None,
        "cpc_sub_patent": None,
        "patent_cpc_main": None,
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
    print("Preparing CPC clusters")
    print(datetime.now())
    
    if loading:

        ffile = open("data/clusters_3.pkl", "rb")
        cpc_time_series = pickle.load(ffile)

    else:

        cpc_time_series, tensors = run_ML(tensors, start, end)
        a_file = open("data/clusters.pkl", "wb")
        pickle.dump(cpc_time_series, a_file)
        a_file.close()
        #print_output("std_out/process/ml_main")

    print("Finished CPC clusters")
    print(datetime.now())

    topical_patents = finding_topical_patents(tensors["patent"], keywords)

    unfold_network(cpc_times_series, tensors, topical_patents)


if __name__ == "__main__":
    start = datetime(1970,1,1)
    end = datetime(2021,12,31)
    words = ["cyber"]
    managerial_layer(start, end, keywords=words)
