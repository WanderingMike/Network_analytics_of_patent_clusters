number_of_cores = 6

tensors = {"assignee":          {"tensor": None,
                                 "dataset": "assignee",
                                 "leading_column": "assignee_id",
                                 "remaining_columns": ["organisation"],
                                 "tensor_value_format": None},
            "cpc_patent":       {"tensor": None,
                                 "dataset": "cpc_current",
                                 "leading_column": "cpc_group",
                                 "remaining_columns": ["patent_id"],
                                 "tensor_value_format": list()},
            "patent_cpc":       {"tensor": None,
                                 "dataset": "cpc_current",
                                 "leading_column": "patent_id",
                                 "remaining_columns": ["cpc_group"],
                                 "tensor_value_format": list()},
            "otherreference":   {"tensor": None,
                                 "dataset": "otherreference",
                                 "leading_column": "patent_id",
                                 "remaining_columns": ["otherreference"],
                                 "tensor_value_format": None},
            "patent":           {"tensor": None,
                                 "dataset": "patent",
                                 "leading_column": "patent_id",
                                 "remaining_columns": ["date", "abstract", "num_claims"],
                                 "tensor_value_format": dict()},
            "patent_assignee":  {"tensor": None,
                                 "dataset": "patent_assignee",
                                 "leading_column": "patent_id",
                                 "remaining_columns": ["assignee_id"],
                                 "tensor_value_format": list()},
            "assignee_patent":  {"tensor": None,
                                 "dataset": "patent_assignee",
                                 "leading_column": "assignee_id",
                                 "remaining_columns": ["patent_id"],
                                 "tensor_value_format": list()},
            "inventor":         {"tensor": None,
                                 "dataset": "patent_inventor",
                                 "leading_column": "patent_id",
                                 "remaining_columns": ["inventors"],
                                 "tensor_value_format": None},
            "forward_citation": {"tensor": None,
                                 "dataset": "uspatentcitation",
                                 "leading_column": "citation_id",
                                 "remaining_columns": ["patent_id"],
                                 "tensor_value_format": list()},
            "backward_citation":{"tensor": None,
                                 "dataset": "uspatentcitation",
                                 "leading_column": "patent_id",
                                 "remaining_columns": ["citation_id"],
                                 "tensor_value_format": list()},
            "year_patent":      {"tensor": None,
                                 "dataset": "patent",
                                 "leading_column": "year",
                                 "remaining_columns": ["patent_id"],
                                 "tensor_value_format": list()}
           }
