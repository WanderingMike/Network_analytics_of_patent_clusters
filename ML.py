import pandas as pd
import numpy as np
from datetime import datetime
import sys
from data_preprocessing import *
from random import shuffle
from functions.functions_ML import *

import blobcity as bc
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score


pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 25)
np.set_printoptions(threshold=sys.maxsize)

number_of_cores = 1


def calculate_emergingness(ml_df, category):
    '''Core of the ML part. This function first divides the data into completed data with pre-existing forward citations
    on the chosen time period, and a subset of the dataframe for which we need to find the citation count. We then trust
    blobcity's AutoAI framework to choose the optimal ML framework for us, including the optimal hyperparameters.'''

    # Categorise output to make it a classification problem
    ml_df["output"] = ml_df["forward_citations"].apply(categorise_output)

    # Splitting dataframe
    data_to_forecast = ml_df[ml_df["forward_citations"] == np.nan]
    data_to_forecast.to_csv("data/unseen/unseen_{}.csv".format(category), index=True)
    data_ready = ml_df[ml_df["forward_citations"] != np.nan]
    data_ready.drop(["date", "forward_citations"], axis=1, inplace=True)

    # Train model
    model = bc.train(df=data_ready, target="output")

    print("Selected model features")
    model.features()
    model.plot_feature_importance()

    model.spill("functions/ML_generation/ML_generation_{}.py".format(category))

    predictions = model.predict(file="data/unseen/unseen_{}.csv".format(category))
    print(predictions)
    model.plot_prediction()
    model.summary()
    model.save("functions/ML_models/ML_model_{}.pkl".format(category))

    # Supplement my_df with new predicted forward citations
    ml_df[ml_df.isnull()] = predictions

    return ml_df


def calculate_indicators(ml_df, start, end, category):
    '''
    This function calculates the tree indicators for one CPC and for every year in the period range:
    - emergingness: the average citation level
    - patent_count: the number of patents at the end of the year
    - citation_count: total citation count
    :param time_series: The squeleton of the three-dimensional dictionary that will be used for the Network Analytics
    :param full_df: Dataframe with values to calculate emergingness
    :param category: CPC group to consider
    :return: returns the time-series, complete for one CPC group
    '''

    indicators = {"emergingness": None,
                  "patent_count": None,
                  "citation_count": None}

    series = {year: indicators for year in range(start.year, end.year + 1)}

    df_final = calculate_emergingness(ml_df, category)

    for year in series.keys():
        temp_df = df_final[ml_df["date"] <= datetime(year, 12, 31)]
        patent_count = len(temp_df.index)
        citation_count = temp_df["forward_citation"].sum()
        emergingness = temp_df["output"].mean()

        series[year]["patent_count"] = patent_count
        series[year]["citation_count"] = citation_count
        series[year]["emergingness"] = emergingness

    return time_series


class Worker(Process):
    '''Each Worker process will create part of the tensor. This tensor (Python dictionary) will have as keys a subset of
    either patent IDs, assignee IDs, or cpc categories. The values will be populated according to the breadth of content
    in each dataframe tensor_df.
    Format:
        {assignee_A: [patent_1, patent_2,...],
         assignee_B: [...],
         ...}
     '''

    def __init__(self, cpc_groups, pid, return_dict, period_start, period_end):
        Process.__init__(self)
        self.cpc_groups = cpc_groups
        self.my_pid = pid
        self.return_dict = return_dict
        self.period_start = period_start
        self.period_end = period_end
        self.tensors = {
            "assignee": None,
            "cpc_patent": None,
            "patent_cpc": None,
            "otherreference": None,
            "patent": None,
            "patent_assignee": None,
            "assignee_patent": None,
            "inventor": None,
            "forward_citation": None,
            "backward_citation": None
        }
        self.time_series = {category: None for category in categories}

    def run(self):
        # tensors["assignee_patent"] = load_tensor("assignee_patent")
        # tensors["patent_assignee"] = load_tensor("patent_assignee")
        # tensors["patent"] = load_tensor("patent")
        # tensors["forward_citation"] = load_tensor("forward_citation")

        for key in self.tensors.keys():
            self.tensors[key] = load_tensor(key)

        for category in self.cpc_groups:
            ml_df = data_preparation(category, self.period_start, self.period_end)
            self.time_series[category] = calculate_indicators(ml_df,
                                                              self.tensors["cpc_patent"],
                                                              self.tensors["patent"],
                                                              self.tensors["forward_citation"],
                                                              self.period_start,
                                                              self.period_end,
                                                              category)

        # return final process-centric time-series
        self.return_dict[self.my_pid] = self.time_series


def prepare_time_series(period_start, period_end):
    '''
    This function creates different processes that each work on a separate subset of CPC clusters. This work includes
    loading the tensors, extracting all indicators thanks to the data_preprocessing.py script, and then applying the
    best-in-class ML algorithms to predict missing citation count values for the most recently published patents.

    :param period_start: datetime value for start of period
    :param period_end: datetime value for end of period
    :return: dictionary with CPC groups as key, and values are time-series of 3 indicators: emergingness, patent count,
    citations.
    '''

    #cpc_tensor = load_tensor("cpc_patent")
    #categories = cpc_tensor.keys()
    categories = ["H04L"]

    shuffle(categories)
    print("There are {} entities".format(len(categories)))

    # Preparing return objects
    process_id = dict()

    # Processes
    no_processes = number_of_cores
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # Split entities amongst processes
    data_split = np.array_split(categories, no_processes)

    # Starting up each thread with its share of the assignees to analyse
    print("Splitting")
    for i in range(len(data_split)):
        p = Worker(data_split[i], i, return_dict, period_start, period_end)
        p.start()
        process_id[i] = p

    # Merging thread
    for i in range(len(process_id)):
        process_id[i].join()

    # Merging all process timeseries into one dictionary
    time_series_final = pd.concat(return_dict.values())

    return time_series_final


if __name__ == "__main__":
    category = "H04L"
    period_start = datetime(1980, 1, 1)
    period_end = datetime(2020, 12, 31)
    df = pd.read_csv("data/dataframes/test2.csv", index_col=0)
    print(df)
    calculate_indicators(df, period_start, period_end, category)
    #time_series = prepare_time_series(period_start, period_end)











#
#
#
#     def build_ann(self, df, cols, classifier = "ann", chained=False):
#         print("Building model")
#         from sklearn.metrics import accuracy_score
#         from sklearn.metrics import average_precision_score
#
#         if "FC10" in cols: year = 2011
#         elif "FC5" in cols: year = 2016
#         else: year = 2018
#
#         df = df[df["patent_date"] < datetime(year=year, month=1,day=1)].reset_index(drop=True)
#
#         if len(cols) > 1:
#             # StratifiedKfoldCrossValidation
#             df["forecast"] = ""
#             for col in cols:
#                 df["forecast"] += df[col].astype(str)
#             z = df["forecast"]
#             df = df.iloc[:, :-1]
#         else:
#             z = df[cols[0]]
#
#         # Decided whether chained or not
#         if chained:   ### maybe change??
#             y = pd.get_dummies(z)
#         else:
#             # Otherwise: build output variables
#             y = pd.DataFrame()
#             if classifier == "ann":
#                 for col in cols:
#                     one_hot = pd.get_dummies(df[col],prefix=col)
#                     y = pd.concat([y, one_hot], axis=1)
#
#                 ### keeping track of output location
#                 for col in cols:
#                     self.divisions[col] = [i for i, x in enumerate(list(y.columns)) if x.startswith(col)]
#             else:
#                 y = df[cols[0]]
#
#
#         # Keeping fixed proportions with StratifiedKFold
#         from sklearn.model_selection import StratifiedKFold
#         skf = StratifiedKFold(n_splits=3, shuffle=True) # change number of splits!!
#         partitions = []
#         for train_index, test_index in skf.split(df, z):
#             partitions.append([list(train_index),list(test_index)])
#
#         # Splitting and standardising
#         from sklearn.preprocessing import StandardScaler
#         sc = StandardScaler()
#
#         if classifier == "ann":
#             # Building ANN
#             model = tf.keras.models.Sequential()
#             model.add(tf.keras.layers.Dense(units=len(df.columns), activation="relu"))
#             model.add(tf.keras.layers.Dense(units=4, activation="relu"))
#             model.add(tf.keras.layers.Dense(units=len(y.columns), activation="sigmoid"))
#             sgd = tf.keras.optimizers.SGD(0.001)
#             model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
#         elif classifier == "dt":
#             from sklearn.tree import DecisionTreeClassifier
#             average_accuracy = []
#             model = DecisionTreeClassifier(criterion="entropy",random_state=0)
#         elif classifier == "svm":
#             from sklearn.svm import SVC
#             model = SVC(kernel = "rbf",random_state=0)
#         else:
#             print("Incorrect model choice")
#             sys.exit()
#
#         df = df.drop(["patent_number","FC3","FC5","FC10","patent_date","cpcs"], axis = 1)
#
#         for iter in range(len(partitions)):
#             # split
#             X_train = df.reindex(index = partitions[iter][0])
#             y_train = y.reindex(index = partitions[iter][0])
#             X_test = df.reindex(index = partitions[iter][1])
#             y_test = y.reindex(index = partitions[iter][1])
#
#             # scaling
#             X_train = sc.fit_transform(X_train)
#             X_test = sc.transform(X_test)
#
#             # Training
#             if classifier == "ann":
#                 model.fit(X_train, y_train, epochs=20)
#                 predict = model.predict(X_test, batch_size=128)
#                 pred_final = pd.DataFrame(columns=y_test.columns, index=y_test.index)
#                 pred_final = pred_final.fillna(0)
#
#                 for place, patent in enumerate(predict):
#                     for forecast in self.divisions.values():
#                         pred_final.iloc[place, patent[forecast[0]:forecast[-1] + 1].argmax() + forecast[0]] = 1
#
#                 ### accuracy score
#                 for col in pred_final.columns:
#                     print(col, accuracy_score(pred_final[col], y_test[col]))
#                     print(average_precision_score(pred_final[col], y_test[col]))
#                     concatenated = np.c_[pred_final[col],y_test[col]]
#                     print(concatenated)
#             else:
#                 model.fit(X_train, y_train)
#                 predict = model.predict(X_test)
#                 concatenated = np.c_[predict, y_test]
#                 total = len(concatenated)
#                 print(concatenated)
#                 for el in concatenated:
#                     if el[0] != el[1]:
#                         total -= 1
#                 average_accuracy.append(100 * total / len(concatenated))
#                 print("{:.2f}%".format(100 * total / len(concatenated)))
#
#         if classifier != "ann":
#             print("AVERAGE ACCURACY: {:.2f}%. Summary:".format(sum(average_accuracy) / len(partitions)))
#             self.accuracies = average_accuracy
#             for rate in average_accuracy:
#                 print(" > > {:.2f}%".format(rate))
#
#
#         ### final model training
#         df = sc.fit_transform(df)
#         if classifier == "ann":
#             model.fit(df,y, epochs=20)
#         else:
#             model.fit(df,y)
#
#         return model, sc
#
#
#
#
#     def emergingness(self,df, sc, model, cols):
#
#         ### Create prediction array
#         def prediction(x, col_index,name):
#             if name == "FC3": year = 2018
#             elif name == "FC5": year = 2016
#             elif name == "FC10": year = 2011
#
#             if x["patent_date"] > datetime(year=year,month=1,day=1):
#                 tmp = x.drop(labels=["patent_number","FC3","FC5","FC10","cpcs","patent_date"])
#                 output = model.predict(sc.transform(tmp.to_numpy().reshape(1, -1)))[0]
#                 if col_index is None:
#                     return output
#                 else:
#                     return output[col_index[0]:col_index[-1]+1].argmax()
#             else:
#                 return x[name]
#
#         print("Predicting patents")
#         if self.divisions:
#             for k,v in self.divisions.items():
#                 df[k] = df.apply(prediction, col_index = v, name=k, axis = 1)
#         else:
#             df[cols[0]] = df.apply(prediction, col_index= None, name=cols[0], axis=1)
#
#
#         ### Get a set of all subgroup IDs
#         index_classes = list()
#         def parsing(x):
#             individual = list()
#             for patent in x:
#                 subclass = patent["cpc_subgroup_id"].split("/")[0]
#                 index_classes.append(subclass)
#                 individual.append(subclass)
#             return list(set(individual))
#
#         df["cpcs"] = df["cpcs"].apply(parsing)
#
#         index_classes = list(set(index_classes))
#         index_classes = [i for i in index_classes if i[0:4] == self.category]
#
#         ### Explode dataframe according to subgroup IDs
#         df = pd.DataFrame({
#             col: np.repeat(df[col].values, df["cpcs"].str.len())
#             for col in df.columns.drop("cpcs")}
#             ).assign(**{"cpcs": np.concatenate(df["cpcs"].values)})[df.columns]
#
#         ### Create final time series emergingness matrix
#         value = {0: 1, 1: 3, 2: 5, 3: 10}
#
#         def output_emerging(name):
#             years = np.arange(2000, 2021, 1)
#             final = pd.DataFrame(index=index_classes, columns=years)
#
#             for subclass in index_classes:
#                 print("Generating class {}".format(subclass))
#                 for year in years:
#                     summation = 0
#                     for k,v in value.items():
#                         cluster = df[(df["patent_date"] >= datetime(year=year,month=1,day=1))
#                                      & (df["patent_date"] <= datetime(year=year,month=12,day=31))
#                                      & (df["cpcs"] == subclass)
#                                      & (df[name] == k)]
#                         total = df[(df["patent_date"] >= datetime(year=year,month=1,day=1))
#                                    & (df["patent_date"] <= datetime(year=year,month=12,day=31))
#                                    & (df["cpcs"] == subclass)]
#                         summation += len(cluster.index) / (len(total.index) if len(total.index != 0) else 1000000) * v
#                     final[year][subclass] = summation
#
#             print(final)
#             ### Normalising the data
#             column_maxes = final.max()
#             df_max = column_maxes.max()
#             if df_max != 0:
#                 final = final / df_max
#             print(final)
#
#             ### Time-series evolution: N - N-1
#             final_diff = final.diff(axis=1)
#             print(final_diff)
#             final_diff.to_excel("output_tables/{}_{}_{}.xlsx".format(name, self.subcategory.replace("/","-"), cols[0]))
#
#         for timeframe in cols:
#             print("Generating final tables")
#             output_emerging(timeframe)
#
#         print("Accuracies: ",self.accuracies)
#
#         return self.accuracies
#
# #######################################################################################################################
#
# def run(name, year_begin, year_end, cols, model, col_analysis, only_run = False):
#     print("#" * 50)
#     print("Starting category {}".format(name))
#
#     start = Analysis(cpc=name)
#
#     if only_run == False:
#         df = start.data(year_begin, year_end, "{}_data_{}_{}".format(name.replace("/","-"), year_begin, year_end))
#         #df = pd.read_pickle("{}_data_{}_{}".format(name.replace("/","-"), year_begin, year_end))
#         df = start.clean(df, name="{}_cleaned_{}_{}".format(name.replace("/","-"), year_begin, year_end))
#     else:
#         df = pd.read_pickle("{}_cleaned_{}_{}".format(name.replace("/","-"), year_begin, year_end))
#
#     ann_model, sc = start.build_ann(df, cols, classifier=model, chained=False)
#     accuracies = start.emergingness(df, sc, ann_model, col_analysis)
#
#     print("\n CATEGORY {} FINISHED".format(name))
#     print("#" * 50)
#
#     return accuracies
#
# if __name__ == "__main__":
#     prep_csv()
#     categories = ["H04L"]
#     categories = [i.replace(" ","") for i in categories]
#     final_accuracies = {k: [] for k in categories}
#     for cat in categories:
#         accuracies = run(cat, 2000, 2020, ["FC3"],"dt", ["FC3"], only_run = False)
#         final_accuracies[cat] += accuracies
#
#     print(final_accuracies)
#
