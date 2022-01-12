import multiprocessing
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import sys
from multiprocessing import Process
import pickle

pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 25)
np.set_printoptions(threshold=sys.maxsize)

### Uploading csv

def prep_csv():
    global other_ref, multiplex, tts_data, patent_cpc

    print("Other references")
    other_ref = pd.read_csv("data/otherreference.csv", header=0, index_col = 0)
    print("Multiplex")
    a_file = open("data/multiplex_full_pickle.pkl", "rb")
    multiplex = pickle.load(a_file)
    print("TTS data")
    tts_data = pd.read_csv("data/tts_data.csv",header=0, dtype={"patent_id":"string"})
    tts_data.set_index(["N"], inplace=True)
    print("CPC Pickle file")
    b_file = open("data/cpc_pickle.pkl", "rb")
    patent_cpc = pickle.load(b_file)


########################################################################################################################
########################################################################################################################

class Worker(Process):

    def __init__(self, other_ref, df, pid, return_dict,category, multiplex, total_strength, patent_cpc):
        Process.__init__(self)
        self.df = df
        self.my_pid = pid
        self.return_dict = return_dict
        self.other_ref = other_ref
        self.category = category
        self.multiplex = multiplex
        self.total_strength = total_strength
        self.patent_cpc = patent_cpc

    def lookup(self, criteria, data, output):

        url = "https://api.patentsview.org/patents/query"

        param = {
            "q": {"_and": criteria},
            "f": data,
             "o":{"per_page":10000}
        }

        resp = requests.post(url, json=param)
        while (resp.status_code != 200):
            print("Sending out another request")
            resp = requests.post(url, json=param)

        json_data = resp.json()

        return json_data[output]

    def run(self):
        print('Running Process {}'.format(self.pid))

        ### Turn patent grant date into usable format
        self.df["patent_date"] = self.df["patent_date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

        ### Forward citations
        def extract_years(x):
            return [datetime.strptime(patent["citedby_patent_date"], "%Y-%m-%d") \
                        if patent["citedby_patent_date"] \
                        else datetime(2050, 1, 1) for patent in x]

        self.df["citedby_patents"] = self.df['citedby_patents'].apply(
            lambda x: extract_years(x) if x[0]["citedby_patent_number"] else [])

        def f(x, num):
            return len([patent for patent in x["citedby_patents"] if patent - x["patent_date"] < timedelta(days=num)])

        for k, v in {"FC3": 1095, "FC5": 1825, "FC10": 3650}.items():
            self.df[k] = self.df.apply(f, num=v, axis=1)

        ### Classification:
        self.df["num_subclass"] = self.df["cpcs"].apply(lambda x: len(x))

        self.df["mainclass"] = self.df["cpcs"].apply(lambda x: list(set([patent["cpc_group_id"] for patent in x])))

        self.df["cited"] = self.df["cited_patents"].apply(lambda x: [patent["cited_patent_number"] for patent in x])

        (self.df["assignees_num_patents"],
         self.df["core_know_how"],
         self.df["peripheral_know_how"],
         self.df["total_tech_strength"],
         self.df["core_strength"],
         self.df["peripheral_strength"]) = zip(*self.df.apply(self.assignee_knowledge, axis=1))

        ### Other references
        self.df["other_ref"] = self.df["patent_number"].apply(lambda x: self.references(x))

        ### Herfindahl
        print("Calculating Herfindahl Index")
        self.df["Herfindal_main"], self.df["Herfindal_sub"] = zip(*self.df.apply(self.Herfindahl, axis=1))

        print('Collapsing Process {}'.format(self.pid))
        self.return_dict[self.my_pid] = self.df

    def assignee_knowledge(self, x):

        if x["assignees"][0]["assignee_sequence"]:
            if x.name % 50 == 0:
                print("> > Process {} # Retrieving assignee information for patent {} ({})".format(self.my_pid, x.name, x["patent_number"]))
            assignees = list()
            patents_counted = list()
            sum, ckh, cts = 0, 0, 0
            for assignee in x["assignees"]:
                assignees.append(assignee["assignee_id"])
                assignee_id = assignee["assignee_id"]

                # Total know-how (TKH)
                if assignee["assignee_total_num_patents"]:
                    sum += int(assignee["assignee_total_num_patents"])

                for patent in self.multiplex[assignee_id]:
                    if any(k in x["mainclass"] for k in self.multiplex[assignee_id][patent]["groups"]) and patent not in patents_counted:
                        cts += self.multiplex[assignee_id][patent]["cit"]
                        ckh += 1
                        patents_counted.append(patent)

            tts = self.total_strength[self.total_strength["assignee_id"].isin(assignees)]["citation_id"].sum()

            return sum, ckh, sum - ckh, tts, cts, tts - cts

        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


    def references(self, x):
        return self.other_ref.c.get(int(x), 0)


    def Herfindahl(self, x):

        if x["cited"]:

            sc1 = list()
            sc2 = list()
            for patent in x["cited"]:
                try:
                    sc1 += self.patent_cpc[patent]["group"]
                    sc2 += self.patent_cpc[patent]["subgroup"]
                except:
                    continue

            count1 = {k: 0 for k in set(sc1)}
            count2 = {k: 0 for k in set(sc2)}

            counter1 = 0
            counter2 = 0
            for patent in x["cited"]:
                try:
                    for category in self.patent_cpc[patent]["group"]:
                        count1[category] += 1
                        counter1 += 1
                    for subcategory in self.patent_cpc[patent]["subgroup"]:
                        count2[subcategory] += 1
                        counter2 += 1
                except:
                    continue

            herfindahl1 = 1
            for v in count1.values():
                herfindahl1 -= (v / counter1) ** 2

            herfindahl2 = 1
            for v in count2.values():
                herfindahl2 -= (v / counter2) ** 2
        else:
            herfindahl1 = np.nan
            herfindahl2 = np.nan

        return round(herfindahl1,5), round(herfindahl2,5)


########################################################################################################################


class Analysis():

    def __init__(self, cpc):
        if len(cpc) == 4:
            self.group_level = "group"
        else:
            self.group_level = "subgroup"
        self.divisions = {}
        self.category = cpc[:4]
        self.subcategory = cpc

    def data(self,start_date, end_date, name=None):

        level = (self.category if self.group_level == "group" else self.subcategory)

        df = pd.DataFrame(columns=['patent_number', 'patent_num_combined_citations', 'patent_date',
                                   'patent_num_claims', 'inventors', 'assignees', 'cited_patents',
                                   'FC3','FC5','FC10','assignees_num_patents','num_subclass','mainclass','age_backward_citation'])

        def download_api(start, end, df):

            page = 1

            while True:

                print("Downloading page {}".format(page))
                tmp = self.download(page, level, start, end)
                print("Amount of patents downloaded: {} (period {})".format(local_patent_count,start))

                df = pd.concat([df, tmp], axis=0)

                page += 1

                if local_patent_count < 10000:
                    break

            return df

        if self.group_level == "group":

            range_list = [year for year in range(start_date,end_date+1,1)]

            for i in range(len(range_list)):
                df = download_api(range_list[i], range_list[i], df)

        else:
            df = download_api(start_date, end_date, df)

        df.index = np.arange(0, len(df))

        print("Download finished")
        if name:
            df.to_pickle(name)
        return df

    def download(self,page, cpc, start_date, end_date):

        global local_patent_count

        url = "https://api.patentsview.org/patents/query"
        data = {
            "q": {
                "_and":[
                    {"cpc_{}_id".format(self.group_level): cpc},
                    {"_gte":{"patent_date":"{}-01-01".format(start_date)}},
                    {"_lte":{"patent_date":"{}-12-31".format(end_date)}}
                    #{"patent_number":[6011788]}
                ]},
            "s": [{"patent_date": "asc"}],
            "f": ["patent_number",
                  "patent_num_combined_citations",
                  "patent_date",
                  "citedby_patent_number",
                  "citedby_patent_date",
                  "cited_patent_number",
                  "cited_patent_date",
                  "patent_num_claims",
                  "assignee_sequence",
                  "assignee_id",
                  "inventor_last_name",
                  "assignee_total_num_patents",
                  "cpc_group_id",
                  "cpc_subgroup_id"],
            "o": {"per_page": 10000, "page": page}
        }

        resp = requests.post(url, json=data)
        json_data = resp.json()
        df = pd.DataFrame(json_data["patents"])
        local_patent_count = json_data["count"]

        return df

    def clean(self, df, name = None):

        print("Cleaning dataframe [...]")
        print(len(df.index))

        ### Assignee indicators
        partitions = 16

        # multiprocessing
        def parallelize(data):
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            data_split = np.array_split(data, partitions)
            tmp_dic = {}
            for i in range(len(data_split)):
                p = Worker(other_ref, data_split[i],i,return_dict,self.category, multiplex, tts_data, patent_cpc)
                p.start()
                tmp_dic[i] = p

            for i in range(len(tmp_dic)):
                tmp_dic[i].join()

            print("Concatenating subresults")

            return pd.concat([v for v in return_dict.values()])

        df = parallelize(df)

        df["assignees"] = df["assignees"].apply(lambda x: len(x) if x[0]["assignee_sequence"] != None else 0)

        for column in ["assignees_num_patents", "core_know_how", "peripheral_know_how", "total_tech_strength", "core_strength", "peripheral_strength"]:
            df[column] = df[column].replace(np.nan, df[column].median())

        # filling herfindahl gaps
        herf_high = df["Herfindal_main"].median()
        herf_low = df["Herfindal_sub"].median()

        df["Herfindal_main"] = df["Herfindal_main"].replace(np.nan, herf_high)
        df["Herfindal_sub"] = df["Herfindal_sub"].replace(np.nan, herf_low)

        ### Number of Inventors
        df["inventors"] = df["inventors"].apply(lambda x: len(x))

        df = df.drop(["citedby_patents"], axis=1)

        ### Median age of Backward citations
        def g(x):

            count = len(x["cited_patents"])
            if x["cited_patents"][0]["cited_patent_number"]:
                time = 0
                for patent in x["cited_patents"]:
                    try:
                        time += (x["patent_date"] - datetime.strptime(patent["cited_patent_date"], "%Y-%m-%d")).days
                    except:
                        time+=0
                        count-=1
                if count == 0:
                    return 0
                return time / (365*count)
            else:
                return None

        df["age_backward_citation"] = df.apply(g, axis=1)


        ### Label Encoding
        from sklearn.preprocessing import LabelEncoder
        diff_labels = list(set([cl for row in df["mainclass"] for cl in row]))
        le = LabelEncoder()
        le.fit(diff_labels)
        df["mainclass"] = df["mainclass"].apply(lambda x: le.transform(x))


        ### OneHotEncoding
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        test = pd.DataFrame(mlb.fit_transform(df['mainclass']), columns=mlb.classes_, index=df.index)
        df = pd.concat([df, test], axis=1, join="inner")

        df.set_index("patent_number")
        df = df.drop(["cited_patents","mainclass","cited"], axis=1)

        ### Categorising Output Variables
        def h(x):

            if x >= 20:
                return 3
            elif 10 <= x <= 19:
                return 2
            elif 2 <= x <= 9:
                return 1
            else:
                return 0

        for col in ["FC3", "FC5", "FC10"]:
            df[col] = df[col].apply(lambda x: h(x))

        # filling missing values
        df["age_backward_citation"] = df["age_backward_citation"].fillna(0) ###possibility for median

        if name:
            df.to_pickle(name)

        return df


    def build_ann(self, df, cols, classifier = "ann", chained=False):
        print("Building model")
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import average_precision_score

        if "FC10" in cols: year = 2011
        elif "FC5" in cols: year = 2016
        else: year = 2018

        df = df[df["patent_date"] < datetime(year=year, month=1,day=1)].reset_index(drop=True)

        if len(cols) > 1:
            # StratifiedKfoldCrossValidation
            df["forecast"] = ""
            for col in cols:
                df["forecast"] += df[col].astype(str)
            z = df["forecast"]
            df = df.iloc[:, :-1]
        else:
            z = df[cols[0]]

        # Decided whether chained or not
        if chained:   ### maybe change??
            y = pd.get_dummies(z)
        else:
            # Otherwise: build output variables
            y = pd.DataFrame()
            if classifier == "ann":
                for col in cols:
                    one_hot = pd.get_dummies(df[col],prefix=col)
                    y = pd.concat([y, one_hot], axis=1)

                ### keeping track of output location
                for col in cols:
                    self.divisions[col] = [i for i, x in enumerate(list(y.columns)) if x.startswith(col)]
            else:
                y = df[cols[0]]


        # Keeping fixed proportions with StratifiedKFold
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True) # change number of splits!!
        partitions = []
        for train_index, test_index in skf.split(df, z):
            partitions.append([list(train_index),list(test_index)])

        # Splitting and standardising
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()

        if classifier == "ann":
            # Building ANN
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(units=len(df.columns), activation="relu"))
            model.add(tf.keras.layers.Dense(units=4, activation="relu"))
            model.add(tf.keras.layers.Dense(units=len(y.columns), activation="sigmoid"))
            sgd = tf.keras.optimizers.SGD(0.001)
            model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
        elif classifier == "dt":
            from sklearn.tree import DecisionTreeClassifier
            average_accuracy = []
            model = DecisionTreeClassifier(criterion="entropy",random_state=0)
        elif classifier == "svm":
            from sklearn.svm import SVC
            model = SVC(kernel = "rbf",random_state=0)
        else:
            print("Incorrect model choice")
            sys.exit()

        df = df.drop(["patent_number","FC3","FC5","FC10","patent_date","cpcs"], axis = 1)

        for iter in range(len(partitions)):
            # split
            X_train = df.reindex(index = partitions[iter][0])
            y_train = y.reindex(index = partitions[iter][0])
            X_test = df.reindex(index = partitions[iter][1])
            y_test = y.reindex(index = partitions[iter][1])

            # scaling
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # Training
            if classifier == "ann":
                model.fit(X_train, y_train, epochs=20)
                predict = model.predict(X_test, batch_size=128)
                pred_final = pd.DataFrame(columns=y_test.columns, index=y_test.index)
                pred_final = pred_final.fillna(0)

                for place, patent in enumerate(predict):
                    for forecast in self.divisions.values():
                        pred_final.iloc[place, patent[forecast[0]:forecast[-1] + 1].argmax() + forecast[0]] = 1

                ### accuracy score
                for col in pred_final.columns:
                    print(col, accuracy_score(pred_final[col], y_test[col]))
                    print(average_precision_score(pred_final[col], y_test[col]))
                    concatenated = np.c_[pred_final[col],y_test[col]]
                    print(concatenated)
            else:
                model.fit(X_train, y_train)
                predict = model.predict(X_test)
                concatenated = np.c_[predict, y_test]
                total = len(concatenated)
                print(concatenated)
                for el in concatenated:
                    if el[0] != el[1]:
                        total -= 1
                average_accuracy.append(100 * total / len(concatenated))
                print("{:.2f}%".format(100 * total / len(concatenated)))

        if classifier != "ann":
            print("AVERAGE ACCURACY: {:.2f}%. Summary:".format(sum(average_accuracy) / len(partitions)))
            self.accuracies = average_accuracy
            for rate in average_accuracy:
                print(" > > {:.2f}%".format(rate))


        ### final model training
        df = sc.fit_transform(df)
        if classifier == "ann":
            model.fit(df,y, epochs=20)
        else:
            model.fit(df,y)

        return model, sc




    def emergingness(self,df, sc, model, cols):

        ### Create prediction array
        def prediction(x, col_index,name):
            if name == "FC3": year = 2018
            elif name == "FC5": year = 2016
            elif name == "FC10": year = 2011

            if x["patent_date"] > datetime(year=year,month=1,day=1):
                tmp = x.drop(labels=["patent_number","FC3","FC5","FC10","cpcs","patent_date"])
                output = model.predict(sc.transform(tmp.to_numpy().reshape(1, -1)))[0]
                if col_index is None:
                    return output
                else:
                    return output[col_index[0]:col_index[-1]+1].argmax()
            else:
                return x[name]

        print("Predicting patents")
        if self.divisions:
            for k,v in self.divisions.items():
                df[k] = df.apply(prediction, col_index = v, name=k, axis = 1)
        else:
            df[cols[0]] = df.apply(prediction, col_index= None, name=cols[0], axis=1)


        ### Get a set of all subgroup IDs
        index_classes = list()
        def parsing(x):
            individual = list()
            for patent in x:
                subclass = patent["cpc_subgroup_id"].split("/")[0]
                index_classes.append(subclass)
                individual.append(subclass)
            return list(set(individual))

        df["cpcs"] = df["cpcs"].apply(parsing)

        index_classes = list(set(index_classes))
        index_classes = [i for i in index_classes if i[0:4] == self.category]

        ### Explode dataframe according to subgroup IDs
        df = pd.DataFrame({
            col: np.repeat(df[col].values, df["cpcs"].str.len())
            for col in df.columns.drop("cpcs")}
            ).assign(**{"cpcs": np.concatenate(df["cpcs"].values)})[df.columns]

        ### Create final time series emergingness matrix
        value = {0: 1, 1: 3, 2: 5, 3: 10}

        def output_emerging(name):
            years = np.arange(2000, 2021, 1)
            final = pd.DataFrame(index=index_classes, columns=years)

            for subclass in index_classes:
                print("Generating class {}".format(subclass))
                for year in years:
                    summation = 0
                    for k,v in value.items():
                        cluster = df[(df["patent_date"] >= datetime(year=year,month=1,day=1))
                                     & (df["patent_date"] <= datetime(year=year,month=12,day=31))
                                     & (df["cpcs"] == subclass)
                                     & (df[name] == k)]
                        total = df[(df["patent_date"] >= datetime(year=year,month=1,day=1))
                                   & (df["patent_date"] <= datetime(year=year,month=12,day=31))
                                   & (df["cpcs"] == subclass)]
                        summation += len(cluster.index) / (len(total.index) if len(total.index != 0) else 1000000) * v
                    final[year][subclass] = summation

            print(final)
            ### Normalising the data
            column_maxes = final.max()
            df_max = column_maxes.max()
            if df_max != 0:
                final = final / df_max
            print(final)

            ### Time-series evolution: N - N-1
            final_diff = final.diff(axis=1)
            print(final_diff)
            final_diff.to_excel("output_tables/{}_{}_{}.xlsx".format(name, self.subcategory.replace("/","-"), cols[0]))

        for timeframe in cols:
            print("Generating final tables")
            output_emerging(timeframe)

        print("Accuracies: ",self.accuracies)

        return self.accuracies

#######################################################################################################################

def run(name, year_begin, year_end, cols, model, col_analysis, only_run = False):
    print("#" * 50)
    print("Starting category {}".format(name))

    start = Analysis(cpc=name)

    if only_run == False:
        df = start.data(year_begin, year_end, "{}_data_{}_{}".format(name.replace("/","-"), year_begin, year_end))
        #df = pd.read_pickle("{}_data_{}_{}".format(name.replace("/","-"), year_begin, year_end))
        df = start.clean(df, name="{}_cleaned_{}_{}".format(name.replace("/","-"), year_begin, year_end))
    else:
        df = pd.read_pickle("{}_cleaned_{}_{}".format(name.replace("/","-"), year_begin, year_end))

    ann_model, sc = start.build_ann(df, cols, classifier=model, chained=False)
    accuracies = start.emergingness(df, sc, ann_model, col_analysis)

    print("\n CATEGORY {} FINISHED".format(name))
    print("#" * 50)

    return accuracies

if __name__ == "__main__":
    prep_csv()
    categories = ["H04L"]
    categories = [i.replace(" ","") for i in categories]
    final_accuracies = {k: [] for k in categories}
    for cat in categories:
        accuracies = run(cat, 2000, 2020, ["FC3"],"dt", ["FC3"], only_run = False)
        final_accuracies[cat] += accuracies

    print(final_accuracies)

