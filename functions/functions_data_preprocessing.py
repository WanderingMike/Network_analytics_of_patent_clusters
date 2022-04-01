from functions.config import *


def fill_date_forward_citations(cluster, tensor_forward_citation, tensor_patent):
    """This function fetches the date and calculates the forward citation count after a
    defined time period. This will become the output variable for the ML algorithm.
    :cluster: ML-readable data-frame
    :tensor_forward_citation: tells us which patents have cited a specific patent
    :tensor_patent: gives us the date of publication of all patents"""

    # Initial time conditions
    years = job_config.prediction_timeframe_years
    period = 365 * years

    def get_forward_citations(row):
        """This function only applies to patents older than the required timeframe. For the remaining patents, the
        forward citation count will be approximated with ML"""

        patent_id = row.name
        patent_date = tensor_patent[patent_id]["date"]

        if patent_date > job_config.data_upload_date - relativedelta(years=years):
            return patent_date, np.nan

        # Get patents that cited patent_id
        citedby_patent_count = 0
        try:
            forward_citations = tensor_forward_citation[patent_id]
        except:
            return patent_date, 0

        # For each retrieved patent, check if it was published within the desired timeframe
        for citedby_patent in forward_citations:
            try:
                if (tensor_patent[citedby_patent]["date"] - patent_date).days < period:
                    citedby_patent_count += 1
            except:
                pass

        return patent_date, citedby_patent_count

    # Applying function to all rows
    cluster["date"], cluster["forward_citations"] = zip(*cluster.apply(get_forward_citations, axis=1))

    return cluster


def fill_cto(cluster, tensor_patent_cpc, tensor_backward_citation):
    """This function calculates the Herfindahl index on all cpc groups of cited patents. The Herfindahl
    index is a measure of how concentrated the groups are. If the index value is close to 0, a patent
    cites other patents part of similar CPC subclasses. If the index is close to 1, they are strewn."""

    def calculate_herfindahl(row):

        # Fetching all cited patents
        patent_id = row.name
        try:
            cited_patents = tensor_backward_citation[patent_id]
        except:
            return 0

        # Variables
        count = {}
        counter = 0

        # Looping through cited patents are recording their cpc classes
        for cited_patent in cited_patents:

            try:
                cpc_subclasses = tensor_patent_cpc[cited_patent]
            except:
                continue  # failure point

            for subclass in cpc_subclasses:
                if subclass in count.keys():
                    count[subclass] += 1
                else:
                    count[subclass] = 1

                counter += 1

        # Calculating index by subtracting squared proportions of cpc groups
        herfindahl_index = 1
        for v in count.values():
            herfindahl_index -= (v / counter) ** 2

        return round(herfindahl_index, 5)

    # Applying to every row
    cluster["CTO"] = cluster.apply(calculate_herfindahl, axis=1)

    return cluster


def fill_pk_tct_tcs(cluster, tensor_backward_citation, tensor_patent):
    """The prior knowledge (PK) index measures the number of backward citations. The Technology Cycle Time (TCT)
    measures the median age of cited patents. The total cited strength (TCS) measures the value of the cited patents."""

    def calculate_pk_tct_tcs(row):

        # Getting all cited patents
        patent_id = row.name
        try:
            cited_patents = tensor_backward_citation[patent_id]
        except:
            return np.nan, np.nan, np.nan

        try:
            mean_cited_fc = cluster[cluster['forward_citations'].isin(cited_patents)]["forward_citations"].mean()
            pk = len(cited_patents)
            tcs = mean_cited_fc
        except:
            pk = 0
            tcs = np.nan

        # For each cited patent, get the age difference
        cited_patents_age = list()
        for cited_patent in cited_patents:
            try:
                age = (tensor_patent[patent_id]["date"] - tensor_patent[cited_patent]["date"]).days
                cited_patents_age.append(age)
            except:
                pass

        # return appropriate value
        try:
            tct = median(cited_patents_age)
        except:
            tct = np.nan

        return pk, tct, tcs

    # Applying to every row
    cluster["PK"], cluster["TCT"], cluster["TCS"] = zip(*cluster.apply(calculate_pk_tct_tcs, axis=1))

    return cluster


def fill_sk(cluster, tensor_otherreference):
    """This function measures the amount of non-patent literature references"""

    # Fetching the value of otherreference for every patent
    def calculate_sk(row):
        try:
            return tensor_otherreference[row.name]
        except:
            return 0

    # Applying to every row
    cluster["SK"] = cluster.apply(calculate_sk, axis=1)

    return cluster


def fill_mf_ts(cluster, tensor_patent_main):
    """This function returns once all the subclasses of a patent, as well as the number of subclasses it is in."""

    def calculate_mf_ts(row):
        try:
            cpc_subclasses = tensor_patent_main[row.name]
            return cpc_subclasses, len(cpc_subclasses)
        except:
            return list(), 0

    # Applying to every row
    cluster["MF"], cluster["TS"] = zip(*cluster.apply(calculate_mf_ts, axis=1))  # $ Main class or all classes? $

    return cluster


def fill_pcd(cluster, tensor_patent):
    """The Protection Coverage measures the amount of claims a patent is making. This information is found
    in the patent tensor"""

    # Function
    def calculate_pcd(row):
        try:
            return tensor_patent[row.name]["num_claims"]
        except:
            return np.nan

    # Applying to every row
    cluster["PCD"] = cluster.apply(calculate_pcd, axis=1)

    return cluster


def fill_col(cluster, tensor_patent_assignee):
    """The collaboration measure is 1 if the patent has more than one assignee, else 0"""

    # Function checks length of assignees list per patent
    def calculate_col(row):
        num_assignees = 0
        try:
            num_assignees = len(tensor_patent_assignee[row.name])
        except:
            pass

        if num_assignees > 1:
            return 1
        else:
            return 0

    # Applying to every row
    cluster["COL"] = cluster.apply(calculate_col, axis=1)

    return cluster


def fill_inv(cluster, tensor_inventor):
    """This functions returns the amount of inventors whose names figure on the patent documents"""

    # Function
    def calculate_inv(row):
        try:
            return tensor_inventor[row.name]
        except:
            return 0

    # Applying to every row
    cluster["INV"] = cluster.apply(calculate_inv, axis=1)

    return cluster


def fill_tkh_ckh_tts_cts(cluster, tensor_patent_assignee, tensor_assignee_patent, tensor_patent_cpc,
                         tensor_forward_citation):
    """This function creates four assignee-related indicators.
    Total Know-How (TKH): number of patents issued by an assignee
    Core Know-How (CKH): number of patents in chosen cpc subgroup issued by an assignee
    Total Technological Strength (TTS): Number of forward citations of patents issued by an assignee
    Core Technological Strength (CTS): Number of forward citations of patents in cpc group issued by an assignee"""

    # Get the four indicators for a specific assignee
    def get_assignee_info(assignee, cpc_classes):
        """Getting all necessary datapoints for one assignee"""

        try:
            assignee_patents = tensor_assignee_patent[assignee]
        except:
            return 0, 0, 0, 0

        assignee_tkh = len(assignee_patents)
        assignee_ckh = 0
        assignee_tts = 0
        assignee_cts = 0
        try:
            set_classes = set(cpc_classes)
        except:
            set_classes = set()

        # Looping through all patents
        for patent in assignee_patents:  # $ do more efficient way? $

            # forward citations
            try:
                forward_citations = len(tensor_forward_citation[patent])
            except:
                forward_citations = 0

            # verify cpc group
            try:
                if bool(set_classes & set(tensor_patent_cpc[patent])):  # $ check here $
                    assignee_ckh += 1
                    assignee_cts += forward_citations
            except:
                pass

            assignee_tts += forward_citations  # $ risk of assignees collision? Create a set of patents?$

        return assignee_tkh, assignee_ckh, assignee_tts, assignee_cts

    def calculate_indices(row):
        """Calculates TKH, CKH, TTS, CTS"""

        # Setting up variables
        patent_id = row.name
        total_know_how = 0
        core_know_how = 0
        total_strength = 0
        core_strength = 0

        # Finding assignees
        try:
            assignee_list = tensor_patent_assignee[patent_id]
        except:
            return np.nan, np.nan, np.nan, np.nan

        # Looping through all assignees
        for assignee in assignee_list:

            if assignee in assignee_info.keys():

                tkh = assignee_info[assignee]["tkh"]
                ckh = assignee_info[assignee]["ckh"]
                tts = assignee_info[assignee]["tts"]
                cts = assignee_info[assignee]["cts"]

            else:

                tkh, ckh, tts, cts = get_assignee_info(assignee, row["MF"])
                assignee_info[assignee] = {"tkh": tkh, "ckh": ckh, "tts": tts, "cts": cts}

            total_know_how += tkh
            core_know_how += ckh
            total_strength += tts
            core_strength += cts

        return total_know_how, core_know_how, total_strength, core_strength

    # Applying to every row
    assignee_info = dict()
    cluster["TKH"], cluster["CKH"], cluster["TTS"], cluster["CTS"] = zip(*cluster.apply(calculate_indices, axis=1))

    return cluster
