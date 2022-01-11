from functions.functions_data_preprocessing import *


def load_tensors(tensors):
    for key in tensors.keys():
        file = open("data/patentsview_cleaned/{}.pkl".format(key), "rb")
        tensors[key] = pickle.load(file)


def generate_dataframe(tensors, category):
    patents_in_cpc_group = tensors["cpc_patent"][category]
    cluster = pd.DataFrame(index = patents_in_cpc_group,
                           columns=['forward_citations', 'CTO', 'STO', 'PK', 'SK', 'TCT', 'MF', 'TS',
                                    'PCD', 'COL', 'INV', 'TKH', 'CKH', 'PKH', 'TTS', 'CTS', 'PTS'])

    return cluster


def fill_dataframe(category, tensors, cluster):

    fill_forward_citations(cluster, tensors["forward_citation"], tensors["patent"])

    fill_cto(cluster)

    fill_pk(cluster, tensors["backward_citation"])

    fill_sk(cluster, tensors["otherreference"])

    fill_tct(cluster)

    fill_mf(cluster, tensors["patent_cpc"])

    fill_ts(cluster, tensors["patent_cpc"])

    fill_pcd(cluster, tensors["patent"])

    fill_col(cluster, tensors["patent_assignee"])

    fill_inv(cluster, tensors["inventor"])

    fill_tkh_ckh_tts_cts(cluster, tensors["patent_assignee"], tensors["assignee_patent"], tensors["cpc_patent"],
                     tensors["forward_citation"], category)

    fill_pkh(cluster)

    fill_pts(cluster)





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


class CPC():

    def __init__(self, cpc):
        self.divisions = {}
        self.category = cpc

    def data(self,start_date, end_date, name=None):

        level = (self.category if self.group_level == "group" else self.subcategory)

        df = pd.DataFrame(columns=['patent_number', 'patent_num_combined_citations', 'patent_date',
                                   'patent_num_claims', 'inventors', 'assignees', 'cited_patents',
                                   'FC3','FC5','FC10','assignees_num_patents','num_subclass','mainclass','age_backward_citation'])



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

def data_preparation(category):
    '''
    1) Load all tensors
    2) Create ML-readable dataframe
    3) Fill in that dataframe
    :param category: CPC group which interests us
    :return: ML-readable dataframe
    '''

    tensors = {"assignee": None,
               "cpc_patent": None,
               "patent_cpc": None,
               "otherreference": None,
               "patent": None,
               "patent_assignee": None,
               "assignee_patent": None,
               "inventor": None,
               "forward_citation": None,
               "backward_citation": None}

    load_tensors(tensors)
    cluster = generate_dataframe(tensors, category)
    cluster_complete = fill_dataframe(category, tensors, cluster)

    return cluster_complete

if __name__ == "__main__":
    data_preparation()



