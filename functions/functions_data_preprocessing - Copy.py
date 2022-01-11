import pandas as pd
import os

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

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

def drop_columns(file, selected_columns = False):
    '''Downsizing data frames by dropping columns, either by preselection or via input
    '''

    original_file_path = "data/patentsview_data/{}.tsv".format(file)

    # Print columns to choose from
    list_file_column_names(original_file_path)

    if not selected_columns:
        selected_columns = input("Which columns should be kept? (Number them) ")
        selected_columns = selected_columns.split()
        selected_columns = [int(element) for element in selected_columns]

    cleaned_df = pd.read_csv(original_file_path, delimiter='\t', usecols=selected_columns)

    return cleaned_df

def clean_patent():
    '''Cleaning patent and assignee information'''

    patent_columns = [0,1] #values might change
    patent_assignee = drop_columns("patent_assignee", selected_columns=patent_columns)
    patent_assignee.columns = ["patent_id", "assignee_id"]

    try:
        patent_assignee.to_csv("data/patentsview_cleaned/patent_assignee.csv")
    except Exception as e:
        print(e.message, e.args)

    return patent_assignee

def clean_uspatentcitation():
    '''Cleaning citation information'''

    uspatentcitation_columns = [1,2]
    uspatentcitation = drop_columns("uspatentcitation", selected_columns=uspatentcitation_columns)

    # Summing up backward citation count per patent
    back_cit = uspatentcitation['patent_id'].value_counts().to_frame()
    back_cit.reset_index(level=0, inplace=True)
    back_cit.columns = ["patent_id", "backward_cit"]

    # Summing up forward citation count per patent
    for_cit = uspatentcitation['citation_id'].value_counts().to_frame()
    for_cit.reset_index(level=0, inplace=True)
    for_cit.columns = ["patent_id", "forward_cit"]

    # Merging the two
    uspatentcitation = back_cit.merge(for_cit, on='patent_id', how='outer')
    print("here", uspatentcitation)

    try:
        uspatentcitation.to_csv("data/patentsview_cleaned/uspatentcitation.csv")
    except Exception as e:
        print(e.message, e.args)

    return uspatentcitation

def clean_cpc_current():
    '''Cleaning cpc information'''

    cpc_current_columns = [1, 4]
    cpc_current = drop_columns("cpc_current", selected_columns=cpc_current_columns)
    cpc_current.columns = ["patent_id", "cpc_group"]

    try:
        cpc_current.to_csv("data/patentsview_cleaned/cpc_current.csv")
    except Exception as e:
        print(e)

    return cpc_current

def clean_otherreference():
    '''Cleaning other references information'''

    otherreference_columns = [1]
    otherreference = drop_columns("otherreference", selected_columns=otherreference_columns)
    otherreference = otherreference['patent_id'].value_counts().to_frame()
    otherreference.reset_index(level=0, inplace=True)
    otherreference.columns = ["patent_id", "other_ref"]

    try:
        otherreference.to_csv("data/patentsview_cleaned/otherreference.csv")
    except Exception as e:
        print(e.message, e.args)

    return otherreference

def multiplex(load = False):
    '''This function concatenates all four datasets into one multiplex with columns:
    patent_id: number of patent, duplicates might appear if there is more than one assignee or CPC category
    assignee_id: assignee of said patent
    citation_count: number of citations the patent received
    cpc: CPC category for the patent. There can be several for each patent'''

    if load:
        print("Reading files")
        patent_assignee = pd.read_csv("data/patentsview_cleaned/patent_assignee.csv", index_col=0)
        print(patent_assignee)
        uspatentcitation = pd.read_csv("data/patentsview_cleaned/uspatentcitation.csv", index_col=0)
        print(uspatentcitation)
        cpc_current = pd.read_csv("data/patentsview_cleaned/cpc_current.csv", index_col=0)
        print(cpc_current)
        otherreference = pd.read_csv("data/patentsview_cleaned/otherreference.csv", index_col=0)
    else:
        print("Cleaning files in real-time")
        patent_assignee = clean_patent()
        uspatentcitation = clean_uspatentcitation()
        cpc_current = clean_cpc_current()
        otherreference = clean_otherreference()

    # Merging files
    multiplex  = patent_assignee.merge(uspatentcitation, on='patent_id', how='left')
    print(multiplex)
    multiplex = multiplex.merge(otherreference, on='patent_id', how='left')
    print(multiplex)
    multiplex = multiplex.merge(cpc_current, on='patent_id', how='left')
    print(multiplex)

    try:
        multiplex.to_csv("data/patentsview_cleaned/multiplex.csv")
        print("Multiplex created.")
    except Exception as e:
        print(e)

