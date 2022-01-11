from functions.functions_data_preprocessing import *
from tensor_deployment import *

if __name__ == "__main__":
    #list_file_column_names("data/patentsview_data/patent_inventor.tsv")
    clean_assignee()
    clean_cpc_current()
    clean_otherreference()
    clean_patent()
    clean_patent_assignee()
    clean_patent_inventor()
    clean_uspatentcitation()