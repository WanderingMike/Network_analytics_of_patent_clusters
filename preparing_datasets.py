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

    def __init__(self, assignees_lst, pid, return_dict, multiplex):
        Process.__init__(self)
        self.assignees_lst = assignees_lst
        self.my_pid = pid
        self.return_dict = return_dict
        self.multiplex = multiplex

    def run(self):

        # Setting up work environment
        print('Running Process {}'.format(self.pid))

        multiplex_subset = self.multiplex[self.multiplex["assignee_id"].isin(self.assignees_lst)]
        print("Filtering done.")

        total = len(multiplex_subset.index)
        count = 0
        answer = {k: {} for k in self.assignees_lst} #creating one entry per assignee

        # Going through entire subset
        for index, row in multiplex_subset.iterrows():

            if (count % 200000 == 0):
                print("> > Process {}: {}/{}".format(self.my_pid,count,total))
            count += 1

            if row["cpc_group"]:
                if row["patent_id"] in answer[row["assignee_id"]]:
                    answer[row["assignee_id"]][row["patent_id"]]["groups"].append(row["cpc_group"])
                    answer[row["assignee_id"]][row["patent_id"]]["cit"] = row["citation_count"]
                else:
                    answer[row["assignee_id"]][row["patent_id"]] = {"cit": row["citation_count"], "groups": [row["cpc_group"]]}

        print('Collapsing Process {}'.format(self.pid))

        self.return_dict[self.my_pid] = answer

def parallelisation(multiplex):
    '''Running the max allowable number of threads to process data for every assignee. Each thread creates for its
    assignees a list of patents, their citation count and cpc classification. The final dictionary will help process
    assignee value faster in the ML and network analysis scripts.'''

    # Setting up environment
    ## Getting all unique assignees
    unique_assignees = multiplex["assignee_id"].unique()
    shuffle(unique_assignees)
    print("There are {} assignees".format(len(unique_assignees)))

    ## Preparing return objects
    tmp_dic = {}
    multiplex_dic = dict()

    ## Processes
    no_processes = 16
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    ## Split assignees amongst processes
    data_split = np.array_split(unique_assignees, no_processes)

    # Starting up each thread with its share of the assignees to analyse
    print("Splitting")
    for i in range(len(data_split)):
        p = Worker(data_split[i], i, return_dict, multiplex)
        p.start()
        tmp_dic[i] = p

    # Merging thread
    for i in range(len(tmp_dic)):
        tmp_dic[i].join()

    print("Merging all thread dictionaries")
    for el in return_dict.values():
        multiplex_dic = dict(multiplex_dic, **el)

    return multiplex_dic

def make_assignee_dictionary():
    '''Importing multiplex: patent_id, assignee_id, value_count, cpc_current
    Running the parallel processing algorithm
    Saving the final multiplex dictionary in a pickle
    '''

    print("Importing multidata")
    multiplex = pd.read_csv("data/patentsview_cleaned/multiplex.csv", nrows = 1000, header=0, index_col=0, dtype={"patent_id": "string",
                                                                                                    "assignee_id":"string",
                                                                                                    "citation_count":float,
                                                                                                    "cpc_group":"string"})
    multiplex["citation_count"] = multiplex["citation_count"].fillna(0)

    multiplex_dic = parallelisation(multiplex)

    # Saving dictionary multiplex as compressed pickle file
    a_file = open("data/patentsview_cleaned/multiplex_dic.pkl", "wb")
    pickle.dump(multiplex_dic, a_file)
    a_file.close()
