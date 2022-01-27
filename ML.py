from data_preprocessing import *
from random import shuffle
from functions.functions_ML import *
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
import os
from datetime import datetime


pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 25)
np.set_printoptions(threshold=sys.maxsize)

number_of_cores = 2


def calculate_emergingness(ml_df, category, clean_files=False):
    '''Core of the ML part. This function first divides the data into completed data with pre-existing forward citations
    on the chosen time period, and a subset of the dataframe for which we need to find the citation count. We then trust
    blobcity's AutoAI framework to choose the optimal ML framework for us, including the optimal hyperparameters.'''

    # Categorise output to make it a classification problem
    ml_df["output"] = ml_df["forward_citations"].apply(categorise_output)
    data_to_forecast = ml_df[ml_df["forward_citations"].isna()]
    X = ml_df[~ml_df["forward_citations"].isna()]

    print(X)
    print(data_to_forecast)

    # Splitting dataframe
    print(datetime.now())

    cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60,
                                                       resampling_strategy='cv', 
                                                       resampling_strategy_arguments={'folds':5},
                                                       memory_limit=None)

    cls.fit(X.drop(["date", "forward_citations", "output"], axis=1), X["output"])
    print(cls.sprint_statistics())
    print(datetime.now())
    predictions = cls.predict(data_to_forecast.drop(["date", "forward_citations", "output"], axis=1))
    print(predictions)
    print(cls.sprint_statistics())

    data_to_forecast["output"] = predictions

    ml_df = pd.concat([X, data_to_forecast], axis=0)

    print(ml_df)
    print(datetime.now())

    return ml_df


def calculate_indicators(ml_df, start, end, category, tensor_patent):
    '''
    This function calculates two indicators and retrieves textual information per CPC group per year:
    - emergingness: the average citation level
    - patent_count: the number of patents at the end of the year
    - keywords: main keywords found in patent abstracts
    - topic: main topics found in patent abstracts
    :param ml_df: data frame used for ML analytics
    :param start: start of time series
    :param end: end of time series
    :param category: CPC group to consider
    :param tensor_patent: contains all patent abstracts
    :return: returns the time-series, complete for one CPC group
    '''

    indicators = {"emergingness": None,
                  "patent_count": None,
                  "keywords": None,
                  "topic": None}

    series = {year: indicators for year in range(start.year, end.year + 1)}

    df_final = calculate_emergingness(ml_df, category)

    for year in series.keys():
        # Filtering patents
        temp_df = df_final[df_final["date"] <= datetime(year, 12, 31)]
        patents_per_year = list(temp_df.index.values)

        # Calculating indicators
        patent_count = len(patents_per_year)
        emergingness = temp_df["output"].mean()

        # Information extraction
        text = ""
        for patent in patents_per_year:
            text += tensor_patent[patent]["abstract"]
            text += " "
        
        try:
            keywords = extract_keywords(text)
        except Exception as e:
            print(e)
            keywords = ["test", "keywords"] # $ chenge $

        try:
            topic = extract_topic(text)
        except Exception as e:
            print(e)
            topic = ["Test topic"]

        print("Finished")

        # Adding to time-series
        series[year]["emergingness"] = emergingness
        series[year]["patent_count"] = patent_count
        series[year]["keywords"] = keywords
        series[year]["topic"] = topic

    return series


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
        self.time_series = {category: None for category in cpc_groups}
        print_output("ml", self.my_pid)

    def run(self):
        
        print(self.cpc_groups)
        for key in self.tensors.keys():
            self.tensors[key] = load_tensor(key)

        for category in self.cpc_groups:
            ml_df = data_preparation(category, self.tensors, self.period_start, self.period_end)
            self.time_series[category] = calculate_indicators(ml_df,
                                                              self.period_start,
                                                              self.period_end,
                                                              category,
                                                              self.tensors["patent"])

        #final process-centric time-series
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

    cpc_tensor = load_tensor("cpc_patent")
    categories = list(cpc_tensor.keys())[:9] # $ delete!! $

    shuffle(categories)
    print("There are {} entities".format(len(categories)))
    print(categories)

    # Preparing return objects
    process_id = dict()
    time_series_final = dict()

    # Processes
    no_processes = number_of_cores
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # Split entities amongst processes
    data_split = np.array_split(categories, no_processes)
    print(os.getpid())

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
    print("Merging CPC cluster dictionaries")
    for process_dictionary in return_dict.values():
        time_series_final = {**time_series_final, **process_dictionary}
    
    print(time_series_final)
    return time_series_final

