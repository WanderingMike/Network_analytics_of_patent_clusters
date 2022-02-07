from datetime import datetime


class MlConfig:

    def __init__(self):
        self.number_of_cores = 2
        self.search_min = 20
        self.search_hours = 0
        self.ml_search_time = self.search_min * 60 + self.search_hours * 3600
        self.size_dataframe = 3000
        self.start = datetime(1970, 1, 1)
        self.end = datetime(2021, 12, 31)
        self.jobs = [["cyber", "honeypot", "cybersecurity"],
                    ["quantum", "computer", "quantum mechanics"],
                    ["5G"]]


job_config = MlConfig()
