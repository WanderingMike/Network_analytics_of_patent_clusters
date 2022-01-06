import pandas as pd
import multiprocessing
from multiprocessing import Process
import numpy as np
import requests
import pickle
from random import shuffle

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def prep_csv():
    global assignee_set, multi_data, cpc_group

    print("Multidata")
    multi_data = pd.read_csv("multi_data.csv",header=0, dtype={"patent_id":"string"})
    multi_data["citation_id"] = multi_data["citation_id"].fillna(0)

    print("Assignees")
    assignee_set = pd.read_csv("patent_assignee_set.csv", header = 0)["assignee_id"].to_list()
    shuffle(assignee_set)

class Worker(Process):

    def __init__(self, lst, pid, return_dict, multi_data):
        Process.__init__(self)
        self.lst = lst
        self.my_pid = pid
        self.return_dict = return_dict
        self.multi_data = multi_data

    def run(self):
        print('Running Process {}'.format(self.pid))
        data = self.multi_data[self.multi_data["assignee_id"].isin(self.lst)]
        print("Filtering done.")
        total = len(data.index)
        count = 0
        answer = {k: {} for k in self.lst}

        # Going through entire subset
        for index, row in data.iterrows():

            if (count % 200000 == 0):
                print("> > Process {}: {}/{}".format(self.my_pid,count,total))
            count += 1

            if row["group_id"]:
                if row["patent_id"] in answer[row["assignee_id"]]:
                    answer[row["assignee_id"]][row["patent_id"]]["groups"].append(row["group_id"])
                    answer[row["assignee_id"]][row["patent_id"]]["cit"] = row["citation_id"]
                else:
                    answer[row["assignee_id"]][row["patent_id"]] = {"cit": row["citation_id"], "groups": [row["group_id"]]}

        print('Collapsing Process {}'.format(self.pid))

        self.return_dict[self.my_pid] = answer

def parallelize():
    print("There are {} assignees".format(len(assignee_set)))
    final = dict()
    partitions = 16
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    data_split = np.array_split(assignee_set, partitions)
    tmp_dic = {}
    print("Splitting")
    for i in range(len(data_split)):
        p = Worker(data_split[i], i, return_dict, multi_data)
        p.start()
        tmp_dic[i] = p

    for i in range(len(tmp_dic)):
        tmp_dic[i].join()

    print("Concatenating subresults")

    for el in return_dict.values():
        final = dict(final, **el)

    return final

if __name__ == '__main__':
    prep_csv()
    df = parallelize()
    a_file = open("multiplex_full_pickle.pkl", "wb")
    pickle.dump(df, a_file)
    a_file.close()