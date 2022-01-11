import pandas as pd
import multiprocessing
from multiprocessing import Process
import numpy as np
import pickle
from random import shuffle

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class Worker(Process):
    '''Each Worker Process will create a dictionary of assignee information. This dictionary will have as keys the unique
    values of a subset of the entire patent assignee list, and as values lists of patents and their respective features.
    Format:

        {assignee_A: [patent_1:{cit:XX, groups: [R,T,Q]},
                      patent_2:{...},
                      ...],
         assignee_B: [...],
         ...}

     '''

    def __init__(self, entities_lst, pid, return_dict, tensor, entity):
        Process.__init__(self)
        self.entities_lst = entities_lst
        self.my_pid = pid
        self.return_dict = return_dict
        self.tensor = tensor
        self.entity = entity
        self.entity_id = "{}_id".format(entity)

    def run(self):
        # Setting up work environment
        print('Running Process {}'.format(self.pid))

        tensor_subset = self.tensor[self.tensor[self.entity_id].isin(self.entities_lst)]
        print("Filtering done.")

        total = len(tensor_subset.index)
        count = 0
        answer = {k: {} for k in self.entities_lst}  # creating one entry per entity

        # Going through entire subset
        if(self.entity == "assignee"):
                answer = self.flatten_assignee(tensor_subset, total, count, answer)
        else:
                answer = self.flatten_patent(tensor_subset, total, count, answer)

        self.return_dict[self.my_pid] = answer
        print('Collapsing Process {}'.format(self.pid))

    def flatten_assignee(self, tensor_subset, count, total, answer):
        '''For each row in the tensor, check if the cpc group exists. The row can usually be discarded if there isn't
        Seconly, for each assignee key in the dictionary, create as a value a list of patents as well as their citation
        count and cpc group'''

        for index, row in tensor_subset.iterrows():

            if (count % 200000 == 0):
                print("> > Process {}: {}/{}".format(self.my_pid,count,total))
            count += 1

            if row["cpc_group"]:

                # if patent is already mentioned in assignee value list
                if row["patent_id"] in answer[row["assignee_id"]]:
                    # update cpc group
                    answer[row["assignee_id"]][row["patent_id"]]["groups"].append(row["cpc_group"])
                    # answer[row["assignee_id"]][row["patent_id"]]["cit"] = row["forward_cit"] $ only forward, not backward? $

                # if patent appears for the first time, initialise it
                else:
                    answer[row["assignee_id"]][row["patent_id"]] = {"back_cit": row["backward_cit"],
                                                                    "for_cit": row["forward_cit"],
                                                                    "groups": [row["cpc_group"]]}

        return answer

    def flatten_patent(self, tensor_subset, count, total, answer):
        '''For each row in the tensor, check if the cpc subgroup exists. The row can usually be discarded if there isn't
        any. Secondly, for each patent key in the dictionary, create as a value a list of assignees, cpc groups and cpc
        subgroups, as well as the backward and forward citation count.'''

        for index, row in tensor_subset.iterrows():

            if (count % 200000 == 0):
                print("> > Process {}: {}/{}".format(self.my_pid,count,total))
            count += 1

            if row["cpc_subgroup"]:

                # if the patent's value is already populated
                if answer[row["patent_id"]]:
                    # update assignee list
                    if row["assignee_id"] not in answer[row["patent_id"]]["assignee"]:
                        answer[row["patent_id"]]["assignee"].append(row["assignee_id"])
                    # update cpc group
                    if row["cpc_group"] not in answer[row["patent_id"]]["cpc_group"]:
                        answer[row["patent_id"]]["cpc_group"].append(row["cpc_group"])
                    # update cpc subgroup, unique everytime since it is at the very right side of the tensor
                    # (result of merging datasets on the left)
                    answer[row["patent_id"]]["cpc_subgroup"].append(row["cpc_subgroup"])

                # if the patent's value is still an empty dictionary, initialise it
                else:
                    answer[row["patent_id"]] = {"assignee":[row["assignee_id"]],
                                                "back_cit": row["backward_cit"],
                                                "for_cit": row["forward_cit"],
                                                "groups": [row["cpc_group"]],
                                                "subgroups":[row["cpc_subgroup"]]}

        return answer

def parallelisation(tensor, entity):
    '''Running the max allowable number of threads to process data for every assignee. Each thread creates for its
    assignees a list of patents, their citation count and cpc classification. The final dictionary will help process
    assignee value faster in the ML and network analysis scripts.'''

    # Setting up environment
    ## Getting all unique assignees
    unique_entities = tensor["{}_id".format(entity)].unique()
    shuffle(unique_entities)
    print("There are {} {}".format(len(unique_entities), entity))

    ## Preparing return objects
    tmp_dic = {}
    multiplex_dic = dict()

    ## Processes
    no_processes = 16
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    ## Split assignees amongst processes
    data_split = np.array_split(unique_entities, no_processes)

    # Starting up each thread with its share of the assignees to analyse
    print("Splitting")
    for i in range(len(data_split)):
        p = Worker(data_split[i], i, return_dict, tensor, entity)
        p.start()
        tmp_dic[i] = p

    # Merging thread
    for i in range(len(tmp_dic)):
        tmp_dic[i].join()

    print("Merging all thread dictionaries")
    for el in return_dict.values():
        tensor_flat = dict(multiplex_dic, **el)

    return tensor_flat

def make_assignee_dictionary():
    '''Importing multiplex: patent_id, assignee_id, value_count, cpc_current
    Running the parallel processing algorithm
    Saving the final multiplex dictionary in a pickle
    '''

    print("Importing multidata")
    tensor = pd.read_csv("data/patentsview_cleaned/multiplex.csv", header=0, index_col=0, dtype={"patent_id": "string",
                                                                                                    "assignee_id":"string",
                                                                                                    "backward_cit":float,
                                                                                                    "forward_cit":float,
                                                                                                    "cpc_group":"string",
                                                                                                    "cpc_subgroup":"string"})
    #tensor["citation_count"] = tensor["citation_count"].fillna(0)

    tensor_assignee_flat = parallelisation(tensor)
    tensor_patent_flat = parallelisation(tensor)

    # Saving dictionary multiplex as compressed pickle file
    a_file = open("data/patentsview_cleaned/tensor_assignee.pkl", "wb")
    pickle.dump(tensor_assignee_flat, a_file)
    a_file.close()

    b_file = open("data/patentsview_cleaned/tensor_patent.pkl", "wb")
    pickle.dump(tensor_patent_flat, b_file)
    b_file.close()
