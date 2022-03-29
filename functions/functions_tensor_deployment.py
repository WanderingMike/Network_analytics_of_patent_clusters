from functions.config import *


def list_file_column_names(file_path):
    '''Lists all columns of a dataframe, as well as a data point for each. This helps choose the relevant columns.
    '''

    file = open(file_path, 'r', encoding='utf8')
    file_name = os.path.basename(file_path)

    first_line = file.readline()
    second_line = file.readline()
    column_names = first_line.split("\t")

    print("File {} has the following columns: \n\t{}\t{}".format(file_name, first_line, second_line))

    file.close()

    return column_names


def drop_columns(file, selected_columns = None, d_type = None):
    '''Downsizing data frames by dropping columns, either by preselection or via input
    '''

    original_file_path = "data/patentsview_data/{}.tsv".format(file)

    # Print columns to choose from
    list_file_column_names(original_file_path)

    if not selected_columns:
        selected_columns = input("Which columns should be kept? (Number them) ")
        selected_columns = selected_columns.split()
        selected_columns = [int(element) for element in selected_columns]

    cleaned_df = pd.read_csv(original_file_path, delimiter='\t', usecols=selected_columns, dtype=d_type)

    return cleaned_df


def clean_assignee():
    '''Cleaning assignee.tsv data'''

    patent_columns = [0, 4]
    assignee_df = drop_columns("assignee",
                               selected_columns=patent_columns,
                               d_type={"patent_id": "string"})
    assignee_df.columns = ["assignee_id", "organisation"]

    return assignee_df


def clean_cpc_current():
    '''Cleaning cpc_current.tsv data'''

    cpc_current_columns = [1, 4, 5]
    cpc_current = drop_columns("cpc_current",
                               selected_columns=cpc_current_columns,
                               d_type={"patent_id": "string"})
    cpc_current.columns = ["patent_id", "cpc_group", "cpc_subgroup"]
    cpc_current.drop_duplicates(inplace=True)

    return cpc_current


def clean_otherreference():
    '''Cleaning otherreference.tsv data'''

    otherreference_columns = [1]
    otherreference = drop_columns("otherreference", selected_columns=otherreference_columns,
                                  d_type={"patent_id": "string"})
    otherreference = otherreference['patent_id'].value_counts().to_frame()
    otherreference.reset_index(level=0, inplace=True)
    otherreference.columns = ["patent_id", "otherreference"]

    return otherreference


def clean_patent():
    '''Cleaning patent.tsv data'''

    patent_columns = [0, 4, 5, 8]
    patent = drop_columns("patent",
                          selected_columns=patent_columns,
                          d_type={"id": "string", "abstract": "string"})
    patent.columns = ["patent_id", "date", "abstract", "num_claims"]
    patent["date"] = pd.to_datetime(patent["date"])
    patent["year"] = patent["date"].apply(lambda x: x.year)

    return patent


def clean_patent_assignee():
    '''Cleaning patent_assignee.tsv data'''

    patent_assignee_columns = [0, 1]
    patent_assignee = drop_columns("patent_assignee",
                                   selected_columns=patent_assignee_columns,
                                   d_type={"patent_id": "string"})
    patent_assignee.columns = ["patent_id", "assignee_id"]

    return patent_assignee


def clean_patent_inventor():
    '''Cleaning patent_inventor.tsv data'''

    patent_inventor_columns = [0]
    patent_inventor = drop_columns("patent_inventor",
                                   selected_columns=patent_inventor_columns,
                                   d_type={"patent_id": "string"})
    patent_inventor = patent_inventor['patent_id'].value_counts().to_frame()
    patent_inventor.reset_index(level=0, inplace=True)
    patent_inventor.columns = ["patent_id", "inventors"]

    return patent_inventor


def clean_uspatentcitation():
    '''Cleaning uspatentcitation.tsv data'''

    tensor_patent_assignee = load_pickle("data/patentsview_cleaned/patent_assignee.pkl")

    uspatentcitation_columns = [1, 2]
    uspatentcitation = drop_columns("uspatentcitation",
                                    selected_columns=uspatentcitation_columns,
                                    d_type={"patent_id": "string", "citation_id": "string"})
    uspatentcitation.columns = ["patent_id", "citation_id"]

    def check_intersection(patent1, patent2):

        try:
            list1 = tensor_patent_assignee[patent1]
            list2 = tensor_patent_assignee[patent2]
            intersection_size = len(set(list1).intersection(list2))
            if intersection_size > 0:
                print(patent1, list1, patent2, list2)
                return True
        except:
            pass

        return False

    for index, row in uspatentcitation.iterrows():
        if check_intersection(row['patent_id'], row['citation_id']):
            uspatentcitation.drop(index, inplace=True)

    return uspatentcitation


dispatch = {
    'assignee': clean_assignee,
    'cpc_current': clean_cpc_current,
    'otherreference': clean_otherreference,
    'patent': clean_patent,
    'patent_assignee': clean_patent_assignee,
    'patent_inventor': clean_patent_inventor,
    'uspatentcitation': clean_uspatentcitation
}
