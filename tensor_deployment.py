import multiprocessing
from multiprocessing import Process
import numpy as np
import pickle
from random import shuffle
from functions.functions_tensor_deployment import *
from functions.config_tensor_deployment import *

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class Worker(Process):
    '''Each Worker process will create part of the tensor. This tensor (Python dictionary) will have as keys a subset of
    either patent IDs, assignee IDs, or cpc categories. The values will be populated according to the breadth of content
    in each dataframe tensor_df.
    Format:
        {assignee_A: [patent_1, patent_2,...],
         assignee_B: [...],
         ...}
     '''

    def __init__(self, entities_lst, pid, return_dict, tensor_df, leading_column, remaining_columns, tensor_value_format):
        Process.__init__(self)
        self.entities_lst = entities_lst
        self.my_pid = pid
        self.return_dict = return_dict
        self.tensor_df = tensor_df
        self.leading_column = leading_column
        self.remaining_cols = remaining_columns
        self.tensor_value_format = tensor_value_format

    def run(self):
        # Setting up work environment
        print('Running Process {}'.format(self.my_pid))

        df_subset = self.tensor_df[self.tensor_df[self.leading_column].isin(self.entities_lst)]
        print("Filtering done on {}".format(self.my_pid))

        self.return_dict[self.my_pid] = self.analysis(df_subset)
        print('Collapsing Process {}'.format(self.my_pid))

    def analysis(self, df_subset):
        '''This function runs different functions according to the format of the final tensor, such that:
        {key: value} -> self.add_unique_value()
        {key: [value, value,...]} -> self.append_values
        {key: {key: value, key: value,...}} -> self.populate_dictionary()
        '''

        total = len(df_subset.index)
        count = 0

        if self.tensor_value_format is None:
            tensor = {k: None for k in self.entities_lst}
            self.add_unique_value(df_subset, tensor, self.remaining_cols, total, count)
        elif type(self.tensor_value_format) is list:
            tensor = {k: [] for k in self.entities_lst}
            self.append_values(df_subset, tensor, self.remaining_cols, total, count)
        else:
            tensor = {k: {col: None for col in self.remaining_cols} for k in self.entities_lst}
            self.populate_dictionary(df_subset, tensor, self.remaining_cols, total, count)

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
            if count % 1000000 == 0:
                print("> > Process {}: {}/{}".format(self.my_pid, count, total))
            count += 1

            try:
                answer[row[self.leading_column]] = row[remaining_cols[0]]
            except Exception as e:
                print("Problem with Process {}:{}-{}".format(self.pid, index, row) + e)

    def append_values(self, df_subset, answer, remaining_cols, total, count):
        '''
        Appends all data points that have the same leading column value into a common list. If there are more than two
        columns in total, the remaining columns form a new dictionary.
        '''

        for index, row in df_subset.iterrows():

            if count % 1000000 == 0:
                print("> > Process {}: {}/{}".format(self.my_pid, count, total))
            count += 1

            try:
                if len(remaining_cols) > 1:
                    tmp = {k: None for k in remaining_cols}
                    for column in remaining_cols:
                        tmp[column] = row[column]
                else:
                    tmp = row[remaining_cols[0]]

                answer[row[self.leading_column]].append(tmp)

            except Exception as e:
                print("Problem with Process {}:{}-{}".format(self.pid, index, row) + e)

        return answer

    def populate_dictionary(self, df_subset, answer, remaining_cols, total, count):
        '''
        For each remaining column, a key: value pair is created
        '''

        for index, row in df_subset.iterrows():
            if count % 1000000 == 0:
                print("> > Process {}: {}/{}".format(self.my_pid, count, total))
            count += 1

            try:
                for column in remaining_cols:
                    answer[row[self.leading_column]][column] = row[column]

            except Exception as e:
                print("Problem with Process {}:{}-{}".format(self.pid, index, row) + e)

        return answer


def parallelisation(tensor_name, dataset, leading_column, remaining_columns, tensor_value_format):
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

    ## Getting all unique entities to build keys
    unique_entities = tensor_df[leading_column].unique()
    shuffle(unique_entities)
    print("There are {} entities".format(len(unique_entities)))

    ## Preparing return objects
    process_id = dict()
    final_tensor = dict()

    ## Processes
    no_processes = number_of_cores
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    ## Split entities amongst processes
    data_split = np.array_split(unique_entities, no_processes)

    # Starting up each thread with its share of the assignees to analyse
    print("Splitting")
    for i in range(len(data_split)):
        p = Worker(data_split[i], i, return_dict, tensor_df, leading_column, remaining_columns, tensor_value_format)
        p.start()
        process_id[i] = p

    # Merging thread
    for i in range(len(process_id)):
        process_id[i].join()

    print("Merging all thread dictionaries")
    for process_dictionary in return_dict.values():
        final_tensor = {**final_tensor, **process_dictionary}

    # Saving tensor as compressed pickle file
    save_tensor(tensor_name, final_tensor)
    print("Tensor saved.")

    return final_tensor


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
        print("*"*30 + "\nBuilding {} tensor\n".format(tensor_name) + "*"*30)
        tensors[tensor_name]["tensor"] = parallelisation(tensor_name,
                                                         value["dataset"],
                                                         value["leading_column"],
                                                         value["remaining_columns"],
                                                         value["tensor_value_format"])


if __name__ == "__main__":
    name = "year_patent"
    value = tensors[name]
    tensor = parallelisation(name, value["dataset"], value["leading_column"], value["tensor_value_format"])
    print(tensor)
    #clean_patent()
    #list_file_column_names("data/patentsview_data/uspatentcitation.tsv")
    #make_tensors()
