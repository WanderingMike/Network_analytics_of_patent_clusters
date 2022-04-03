from functions.config import *


def fill_date_forward_citations(row, tensor_forward_citation, tensor_patent):
    """This function fetches the date and calculates the forward citation count after a
    defined time period. This will become the output variable for the ML algorithm.
    :cluster: ML-readable data-frame
    :tensor_forward_citation: tells us which patents have cited a specific patent
    :tensor_patent: gives us the date of publication of all patents"""

    years = job_config.prediction_timeframe_years
    period = 365 * years

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


def fill_cto(row, tensor_patent_cpc, tensor_backward_citation):
    """This function calculates the Herfindahl index on all cpc groups of cited patents. The Herfindahl
    index is a measure of how concentrated the groups are. If the index value is close to 0, a patent
    cites other patents part of similar CPC subclasses. If the index is close to 1, they are strewn."""

    print_val = False
    # Fetching all cited patents
    patent_id = row.name
    try:
        cited_patents = tensor_backward_citation[patent_id]
        if len(cited_patents) < 10 and (cited_patents.count("4662438") >= 1 or cited_patents.count("6713728")):
            print("#"*50)
            print(cited_patents)
            print_val = True
    except:
        return 0

    # Variables
    count = {}
    counter = 0

    # Looping through cited patents are recording their cpc classes
    for cited_patent in cited_patents:

        try:
            cpc_subclasses = tensor_patent_cpc[cited_patent]
            show_value(print_val, [cited_patent, cpc_subclasses])
        except:
            continue  # failure point

        for subclass in cpc_subclasses:
            if subclass in count.keys():
                count[subclass] += 1
            else:
                count[subclass] = 1

            counter += 1

    show_value(print_val,[counter, count])
    # Calculating index by subtracting squared proportions of cpc groups
    herfindahl_index = 1
    for v in count.values():
        herfindahl_index -= (v / counter) ** 2

    show_value(print_val, herfindahl_index)
    return round(herfindahl_index, 5)


def fill_pk_tct_tcs(row, df_citations, tensor_backward_citation, tensor_patent):
    """The prior knowledge (PK) index measures the number of backward citations. The Technology Cycle Time (TCT)
    measures the median age of cited patents. The total cited strength (TCS) measures the value of the cited patents."""

    # Getting all cited patents
    patent_id = row.name
    print_val = False
    try:
        cited_patents = tensor_backward_citation[patent_id]
        if len(cited_patents) < 10 and (cited_patents.count("4662438") >= 1 or cited_patents.count("6713728") >= 1):
            print("#"*50)
            print(cited_patents)
            print_val = True
    except:
        return np.nan, np.nan, np.nan

    try:
        mean_cited_fc = df_citations.filter(items=cited_patents)["forward_citations"].mean()
        print_val = True
        show_value(print_val, [df_citations.filter(items=cited_patents)])
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
            show_value(print_val, [tensor_patent[patent_id]["date"], tensor_patent[cited_patent]["date"], age])
            cited_patents_age.append(age)
        except:
            pass

    # return appropriate value
    try:
        tct = median(cited_patents_age)
        show_value(print_val, tct)
    except:
        tct = np.nan

    return pk, tct, tcs


def fill_sk(row, tensor_otherreference):
    """This function measures the amount of non-patent literature references"""
    try:
        return tensor_otherreference[row.name]
    except:
        return 0


def fill_mf_ts(row, tensor_patent_main):
    """This function returns once all the subclasses of a patent, as well as the number of subclasses it is in."""

    try:
        cpc_subclasses = tensor_patent_main[row.name]
        return cpc_subclasses, len(cpc_subclasses)
    except:
        return list(), 0


def fill_pcd(row, tensor_patent):
    """The Protection Coverage measures the amount of claims a patent is making. This information is found
    in the patent tensor"""

    try:
        return tensor_patent[row.name]["num_claims"]
    except:
        return np.nan


def fill_inv(row, tensor_inventor):
    """This functions returns the amount of inventors whose names figure on the patent documents"""

    try:
        return tensor_inventor[row.name]
    except:
        return 0


def fill_col_tkh_ckh_tts_cts(cluster, tensor_patent_assignee, tensor_assignee_patent, tensor_patent_cpc,
                             tensor_forward_citation):
    """
    This function creates five assignee-related indicators.
    Collaboration (COL): 1 if the patent has more than one assignee, else 0
    Total Know-How (TKH): number of patents issued by an assignee
    Core Know-How (CKH): number of patents in chosen cpc subgroup issued by an assignee
    Total Technological Strength (TTS): Number of forward citations of patents issued by an assignee
    Core Technological Strength (CTS): Number of forward citations of patents in cpc group issued by an assignee
    """
    print_val = False
    # Get the four indicators for a specific assignee
    def get_assignee_info(assignee, cpc_classes):
        """Getting all necessary data points for one assignee"""

        try:
            assignee_patents = tensor_assignee_patent[assignee]
            if len(assignee_patents) < 20 and len(assignee_patents) > 6:
                print_val = True
                
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

            assignee_tts += forward_citations  # $ risk of assignees collision? Create a set of patents? $
            show_value(print_val, [assignee, forward_citations, set_classes, tensor_patent_cpc[patent], bool(set_classes & set(tensor_patent_cpc[patent]))])

        show_value(print_val, [assignee_tkh, assignee_ckh, assignee_tts, assignee_cts])
        return assignee_tkh, assignee_ckh, assignee_tts, assignee_cts

    def calculate_indices(row):
        """Calculates COL, TKH, CKH, TTS, CTS"""

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
            return 0, np.nan, np.nan, np.nan, np.nan

        if len(assignee_list) > 1:
            col = 1
        else:
            col = 0

        # Looping through all assignees
        for assignee in assignee_list:

            if assignee in assignee_info.keys():
                if assignee_info[assignee]["tkh"] == 5:
                    print("Found one assignee with 5 tkh!")
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

        return col, total_know_how, core_know_how, total_strength, core_strength

    # Applying to every row
    assignee_info = dict()
    cluster["COL"], cluster["TKH"], cluster["CKH"], cluster["TTS"], cluster["CTS"] = \
        zip(*cluster.apply(calculate_indices, axis=1))

    return cluster
