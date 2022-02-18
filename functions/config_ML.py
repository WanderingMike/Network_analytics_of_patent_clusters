from datetime import datetime


class MlConfig:

    def __init__(self):
        self.search_min = 0
        self.search_hours = 6
        self.ml_search_time = self.search_min * 60 + self.search_hours * 3600
        self.size_dataframe = 15000
        self.upload_date = datetime(2021, 6, 30)
        self.load_main = True
        self.load_df_final = True
        self.load_classifier = True
        self.load_df_filled = True
        self.jobs = [['attack signature',
                    'blue team',
                    'bot',
                    'botnet',
                    'bug',
                    'ciphertext',
                    'computer forensics',
                    'computer security incident',
                    'computer virus',
                    'cryptanalysis',
                    'cryptography',
                    'cryptology',
                    'cyber incidence',
                    'cybersecurity',
                    'cyberspace',
                    'cyber threat intelligence',
                    'data breach',
                    'decryption',
                    'denial of service',
                    'encrypt',
                    'exploit',
                    'firewall',
                    'hacker',
                    'hashing',
                    'keylogger',
                    'malware',
                    'pen test', 
                    'pentest',
                    'phishing',
                    'public key',
                    'red team',
                    'rootkit',
                    'spoofing',
                    'spyware',
                    'threat actor']]


job_config = MlConfig()
