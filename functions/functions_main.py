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
                extracted_token = abstract_tokenised[word_loc:word_loc + len(concept)]
                if ' '.join(extracted_token) == ' '.join(concept):
                    references_count += 1

    return 100 * references_count / len(abstract_tokenised)  # abstract is ~400 words, 1-2 cyberwords: ~1/100


def finding_topical_patents(tensor_patent, keywords):
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
            if reference_count != 0:
                topical_patents[patent] = reference_count
        except:
            continue

    return topical_patents


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
