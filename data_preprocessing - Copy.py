from functions.functions_data_preprocessing import *
from tensor_deployment import *

if __name__ == "__main__":

    #clean_cpc_current()
    #clean_uspatentcitation()
    # Getting assignee information
    #multiplex(load=True)
    #make_assignee_dictionary()

    # cpc
    list_file_column_names("data/patentsview_data/patent.tsv")
    #list_file_column_names("data/tts_data.csv")