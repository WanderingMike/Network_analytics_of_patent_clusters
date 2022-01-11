import multiprocessing
from multiprocessing import Process
import numpy as np
import pickle
from random import shuffle
from functions.functions_data_preprocessing import *

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

tensors = {"assignee": {"tensor": None,
                            "dataset": "assignee",
                            "leading_column": "assignee_id",
                            "tensor_value_format": None},
               "cpc_patent": {"tensor": None,
                              "dataset": "cpc_current",
                              "leading_column": "cpc_group",
                              "tensor_value_format": list()},
               "patent_cpc": {"tensor": None,
                              "dataset": "cpc_current",
                              "leading_column": "patent_id",
                              "tensor_value_format": list()},
               "otherreference": {"tensor": None,
                                  "dataset": "otherreference",
                                  "leading_column": "patent_id",
                                  "tensor_value_format": None},
               "patent": {"tensor": None,
                          "dataset": "patent",
                          "leading_column": "patent_id",
                          "tensor_value_format": {"date": None, "num_claims": None, "abstract": None}},
               "patent_assignee": {"tensor": None,
                                   "dataset": "patent_assignee",
                                   "leading_column": "patent_id",
                                   "tensor_value_format": list()},
               "assignee_patent": {"tensor": None,
                                   "dataset": "patent_assignee",
                                   "leading_column": "assignee_id",
                                   "tensor_value_format": list()},
               "inventor": {"tensor": None,
                            "dataset": "patent_inventor",
                            "leading_column": "patent_id",
                            "tensor_value_format": None},
               "forward_citation": {"tensor": None,
                                    "dataset": "uspatentcitation",
                                    "leading_column": "citation_id",
                                    "tensor_value_format": [{"patent_id": None, "date": None}]},
               "backward_citation": {"tensor": None,
                                     "dataset": "uspatentcitation",
                                     "leading_column": "patent_id",
                                     "tensor_value_format": [{"citation_id": None, "date": None}]}}

class Worker(Process):
    '''Each Worker process will create part of the tensor. This tensor (Python dictionary) will have as keys a subset of
    either patent IDs, assignee IDs, or cpc categories. The values will be populated according to the breadth of content
    in each dataframe tensor_df.
    Format:
        {assignee_A: [patent_1, patent_2,...],
         assignee_B: [...],
         ...}
     '''

    def __init__(self, entities_lst, pid, return_dict, tensor_df, leading_column, tensor_value_format):
        Process.__init__(self)
        self.entities_lst = entities_lst
        self.my_pid = pid
        self.return_dict = return_dict
        self.tensor_df = tensor_df
        self.leading_column = leading_column
        self.tensor_value_format = tensor_value_format

    def run(self):
        # Setting up work environment
        print('Running Process {}'.format(self.pid))

        df_subset = self.tensor_df[self.tensor_df[self.leading_column].isin(self.entities_lst)]
        print("Filtering done.")

        self.return_dict[self.my_pid] = self.analysis(df_subset)
        print('Collapsing Process {}'.format(self.pid))

    def analysis(self, df_subset):
        '''This function runs different functions according to the format of the final tensor, such that:
        {key: value} -> self.add_unique_value()
        {key: [value, value,...]} -> self.append_values
        {key: {key: value, key: value,...}} -> self.populate_dictionary()
        '''

        remaining_cols = list(set(list(df_subset.columns)) - set(list(self.leading_column)))
        tensor = {k: self.tensor_value_format for k in self.entities_lst}
        total = len(df_subset.index)
        count = 0

        if not self.tensor_value_format:
            self.add_unique_value(df_subset, tensor, remaining_cols, total, count)
        elif type(self.tensor_value_format) is list:
            self.append_values(df_subset, tensor, remaining_cols, total, count)
        else:
            self.populate_dictionary(df_subset, tensor, remaining_cols, total, count)

        return tensor

    def add_unique_value(self, df_subset, answer, remaining_cols, total, count):
        '''
        Adds data point in second column as a value in dictionary
        :param df_subset: The dataframe that will be run through
        :param answer: tensor
        :param total: Total amount of rows
        :param count: Row count variable
        '''

        for index, row in df_subset.iterrows():
            if (count % 200000 == 0):
                print("> > Process {}: {}/{}".format(self.my_pid, count, total))
            count += 1

            answer[row[self.leading_column]] = row[remaining_cols[0]]

    def append_values(self, df_subset, answer, remaining_cols, total, count):
        '''
        Appends all data points that have the same leading column value into a common list. If there are more than two
        columns in total, the remaining columns form a new dictionary.
        '''

        for index, row in df_subset.iterrows():
            if count % 200000 == 0:
                print("> > Process {}: {}/{}".format(self.my_pid, count, total))
            count += 1

            if len(remaining_cols) > 1:
                tmp = {j: None for j in remaining_cols}
                for column in remaining_cols:
                    tmp[column] = row[column]
            else:
                tmp = row[remaining_cols[0]]

            answer[row[self.leading_column]].append(tmp)

        return answer

    def populate_dictionary(self, df_subset, answer, remaining_cols, total, count):
        '''
        For each remaining column, a key: value pair is created
        '''

        for index, row in df_subset.iterrows():
            if (count % 200000 == 0):
                print("> > Process {}: {}/{}".format(self.my_pid, count, total))
            count += 1

            for column in remaining_cols:
                answer[row[self.leading_column]][column] = row[column]

        return answer

def parallelisation(tensor_name, dataset, leading_column, tensor_value_format):
    '''
    We want to build a tensor for each of our dataframes. Step-by-step run-through
    1) Load appropriate dataframe
    2) Keys in final tensor are the unique values of the leading column
    3) Distribute these values amongst X different processes
    4) Launch process
    5) Join return values
    :param dataset: Tells us which Patentsview tsv dataset to clean and load
    :param leading_column: This will form the keys of the tensor (basically the first axis of a multi-dim. tensor)
    :param tensor_value_format: Each tensor has a different format (different dimensions), this specifies it
    :return: One of 10 possible tensors
    '''

    # Setting up environment
    ## Loading correct dataset
    tensor_df = dispatch[dataset]()
    print(tensor_df)
    print(leading_column)

    ## Getting all unique entities to build keys
    unique_entities = tensor_df[leading_column].unique()
    shuffle(unique_entities)
    print("There are {} entities".format(len(unique_entities)))

    ## Preparing return objects
    tmp_dic = {}
    multiplex_dic = dict()

    ## Processes
    no_processes = 2
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    ## Split entities amongst processes
    data_split = np.array_split(unique_entities, no_processes)

    # Starting up each thread with its share of the assignees to analyse
    print("Splitting")
    for i in range(len(data_split)):
        p = Worker(data_split[i], i, return_dict, tensor_df, leading_column, tensor_value_format)
        p.start()
        tmp_dic[i] = p

    # Merging thread
    for i in range(len(tmp_dic)):
        tmp_dic[i].join()

    print("Merging all thread dictionaries")
    for el in return_dict.values():
        tensor = dict(multiplex_dic, **el)

    # Saving tensor as compressed pickle file
    save_tensor(tensor_name, tensor)

    return tensor

def save_tensor(tensor_name, tensor):
    a_file = open("data/patentsview_cleaned/{}.pkl".format(tensor_name), "wb")
    pickle.dump(tensor, a_file)
    a_file.close()

def make_tensors():
    '''Importing multiplex: patent_id, assignee_id, value_count, cpc_current
    Running the parallel processing algorithm
    Saving the final multiplex dictionary in a pickle
    '''

    for tensor_name, value in tensors.items():
        print("*"*30+"\nBuilding {} tensor\n".format(tensor_name) + "*"*30)
        tensors[tensor_name]["tensor"] = parallelisation(tensor_name, value["dataset"], value["leading_column"], value["tensor_value_format"])

if __name__ == "__main__":
    make_tensors()
