Network Dynamics of Patent Clusters
=============================================

## Goal

TechRank aims to help decision-makers to quantitatively assess the influence of the entities in order to take relevant investments decisions under high level of uncertainty. 

------
## Abstract
The cybersecurity technological landscape is a complex ecosystem in which entities -- such as  companies and technologies -- influence each other in a non-trivial manner. Understanding influence measures of each entity is central when it comes to  take informed technological  investment  decisions. 

To recognize the mutual influence of companies and technologies in cybersecurity, we consider a bi-partite graph that links companies and technologies. Then, we weight nodes by applying a recursive algorithm based on the method of reflection. This endeavour helps to assign a measure of how an entity impacts the cybersecurity market. Our results help (i) to measure the magnitude of influence of each entity, (ii) decision-makers to address more informed investment strategies, according to their preferences. 

Investors can customise the algorithm by indicating which external factors --such as previous investments and geographical positions-- are relevant for them. They can select their interests among a list of properties about companies and technologies and weights them according to their needs. This preferences are automatically included in the algorithm and the TechRank's scores changes accordingly.

------
## Documents
For more information please refer to [our paper](../docs/_static/TechRank_shortpaper.pdf). Mathematical formulas are substantiated and the system is applied to a use case in cybersecurity technology monitoring.


------
## Code

The code comes in the form of `py` files: 
- the _py files_ contain the function.

**Data**:
[Patentsview](https://www.patentsview.org).
The data is freely available from the bulk data download page (https://patentsview.org/download/data-download-tables)

**Classes** :
We work with 3 dataclasses: `Companies`, `Technologies` and `Investors`.


------
## Files description:

Short description of the files:

| File name        | Short Description  |  
| ------------- |:-------------:| 
| data_preprocessing.py                   	| Creates Machine-Learning inputs from the dictionaries created in tensor_deployment.py |
| main.py					| Runs the entire system and processes high-level system outputs |
| ML.py						| Applies classification algorithms to determine forward citation count & applies NLP to abstracts |
| network_analysis.py				| Builds the network of CPC clusters and assignees |
| tensor_deployment.py				| Loads and flattens data frames into lightweight dictionaries |
| functions/config.py				| Configuration file for tensor formatting |
| functions/functions_data_preprocessing.py	| Contains functions that compute the indicators for the ML algorithms |
| functions/functions_main.py			| Contains functions to find topical patents and create ranked lists of technologies and companies  |
| functions/functions_ML.py			| Contains the NLP keyword extraction and topic modeling functions |
| functions/functions_network_analysis.py	| Contains network-construction functions |
| functions/functions_tensor_deployment.py 	| Contains functions to load tsv files and reduce their size |



Each file contains more details and comments. 

------
## Data sources:

Short description of the files:

| File name        | Short Description  | Columns |  Last updated |
| ------------- |:-------------:|-------------:|-------------:|
| application.tsv | Information on the applications for granted patent | id, patent_id, series_code, number, country, date |
| assignee.tsv | Disambiguated assignee data for granted patents and pre-granted applications | id, type, name_first, name_last, organization | 08.10.2021 |
| cpc_current.tsv | Current CPC classification data for all patents | uuid, patent_id, section_id, subsection_id, group_id, subgroup_id, category, sequence | 08.10.2021 |
| cpc_subgroup.tsv | CPC subgroup names | id, title |
| otherreference.tsv | Non-patent citations mentioned in patents (e.g. articles, papers, etc.) | uuid, patent_id, text, sequence | 08.10.2021 |
| patent.tsv | Data on granted patents | id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn | 08.10.2021 |
| patent_assignee.tsv | Metadata table for many-to-many relationships between patents and assignees | patent_id, assignee_id, location_id | 08.10.2021 |
| patent_inventor.tsv | Metadata table for many-to-many relationships between patents and inventors | patent_id, inventor_id, location_id | 08.10.2021 |
| uspatentcitation.tsv | Citations made to US granted patents by US patents | uuid, patent_id, citation_id, date, name, kind, country, category, sequence | (08.10.2021) |

------
## Hints of bibliography:
The main *sources* of this work are the following:
