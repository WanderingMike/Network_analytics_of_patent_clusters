from functions.functions_tensor_deployment import *


class Worker(Process):
    """Each Worker process will create part of the tensor. This tensor (Python dictionary) will have as keys a subset of
    either patent IDs, assignee IDs, or cpc categories. The values will be populated according to the breadth of content
    in each dataframe pandas_df.
    Format:
        {assignee_A: [patent_1, patent_2,...],
         assignee_B: [...],
         ...}
     """

    def __init__(self, entities_lst, pid, return_dict, pandas_df, leading_column, remaining_columns, tensor_val_format):
        Process.__init__(self)
        self.entities_lst = entities_lst
        self.my_pid = pid
        self.return_dict = return_dict
        self.pandas_df = pandas_df
        self.leading_column = leading_column
        self.remaining_cols = remaining_columns
        self.tensor_value_format = tensor_val_format

    def run(self):
        # Setting up work environment
        print('Running Process {}'.format(self.my_pid))

        df_subset = self.pandas_df[self.pandas_df[self.leading_column].isin(self.entities_lst)]
        print("Filtering done on {}".format(self.my_pid))

        self.return_dict[self.my_pid] = self.analysis(df_subset)
        print('Collapsing Process {}'.format(self.my_pid))

    def analysis(self, df_subset):
        """
        This function runs different functions according to the format of the final tensor, such that:
        {key: value} -> self.populate_keys()
        {key: [value, value,...]} -> self.populate_list()
        {key: {key: value, key: value,...}} -> self.populate_dictionary()
        """

        total = len(df_subset.index)
        count = 0

        if self.tensor_value_format is None:
            tensor = {k: None for k in self.entities_lst}
            self.populate_keys(df_subset, tensor, self.remaining_cols, total, count)
        elif type(self.tensor_value_format) is list:
            tensor = {k: [] for k in self.entities_lst}
            self.populate_list(df_subset, tensor, self.remaining_cols, total, count)
        else:
            tensor = {k: {col: None for col in self.remaining_cols} for k in self.entities_lst}
            self.populate_dictionary(df_subset, tensor, self.remaining_cols, total, count)

        return tensor

    def populate_keys(self, df_subset, answer, remaining_cols, total, count):
        """
        Adds the data point in second column as a value in the dictionary
        :param df_subset: The dataframe that will be run through
        :param answer: tensor
        :param remaining_cols: columns that will form the keys of the dictionary
        :param total: Total amount of rows
        :param count: Row count variable
        """

        for index, row in df_subset.iterrows():
            if count % 1000000 == 0:
                print("> > Process {}: {}/{}".format(self.my_pid, count, total))
            count += 1

            try:
                answer[row[self.leading_column]] = row[remaining_cols[0]]
            except Exception as e:
                print("Problem with Process {}:{}-{}".format(self.pid, index, row) + e)

    def populate_list(self, df_subset, answer, remaining_cols, total, count):
        """
        Appends all data points that have the same leading column value into a common list. If there are more than two
        columns in total, the remaining columns form a new dictionary.
        """

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
        """For each remaining column, a key: value pair is created"""

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
    """
    We want to build a tensor for each of our dataframes. Step-by-step run-through
    1) Load appropriate dataframe
    2) Keys in final tensor are the unique values of the leading column
    3) Distribute these values amongst X different processes
    4) Launch process
    5) Join return values
    :param tensor_name: name of data tensor that is needed
    :param dataset: Tells us which Patentsview tsv dataset to clean and load
    :param leading_column: This will form the keys of the tensor (basically the first axis of a multi-dim. tensor)
    :param remaining_columns: data points that will form the values of the dictionary
    :param tensor_value_format: Each tensor has a different format (different dimensions), this specifies it
    :return: One of 10 possible tensors
    """

    # Setting up environment
    # # Loading correct dataset
    pandas_df = dispatch[dataset]()

    # # Getting all unique entities to build keys
    unique_key_entities = pandas_df[leading_column].unique()
    shuffle(unique_key_entities)
    print("There are {} entities".format(len(unique_key_entities)))

    # # Preparing return objects
    process_table = dict()
    final_tensor = dict()

    # # Processes
    no_computational_cores = job_config.number_of_cores
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # # Split entities amongst processes
    split_data = np.array_split(unique_key_entities, no_computational_cores)

    # Starting up each thread with its share of the assignees to analyse
    print("Splitting")
    for pid in range(len(split_data)):
        p = Worker(split_data[pid], pid, return_dict, pandas_df, leading_column, remaining_columns, tensor_value_format)
        p.start()
        process_table[pid] = p

    # Merging thread
    for pid in range(len(process_table)):
        process_table[pid].join()

    print("Merging all thread dictionaries")
    for process_dictionary in return_dict.values():
        final_tensor = {**final_tensor, **process_dictionary}

    # Saving tensor as compressed pickle file
    save_pickle("data/tensors/{}.pkl".format(tensor_name), final_tensor)
    print("Tensor saved with {} entities.".format(len(final_tensor.keys())))

    return final_tensor


def make_tensors():
    """
    Importing multiplex: patent_id, assignee_id, value_count, cpc_current
    Running the parallel processing algorithm
    Saving the final multiplex dictionary in a pickle
    """

    for tensor_name, tensor_config in tensors_config.items():
        print("*"*30 + "\nBuilding {} tensor\n".format(tensor_name) + "*"*30)
        tensors_config[tensor_name]["tensor"] = parallelisation(tensor_name,
                                                                tensor_config["dataset"],
                                                                tensor_config["leading_column"],
                                                                tensor_config["remaining_columns"],
                                                                tensor_config["tensor_value_format"])


if __name__ == "__main__":
    name = "cpc_sub_patent"
    config = tensors_config[name]
    single_tensor = parallelisation(name,
                                    config["dataset"],
                                    config["leading_column"],
                                    config["remaining_columns"],
                                    config["tensor_value_format"])
    try:
        print(single_tensor["G01S7/4914"])
    except:
        pass
    try:
        print(single_tensor["10000000"])
    except:
        print("Hello")
    # clean_patent()
    # list_file_column_names("data/patentsview_data/uspatentcitation.tsv")
    # make_tensors()
