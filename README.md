Network Dynamics of Patent Clusters
=============================================

## Goal

This library and its associated paper help decision-makers identify emerging technologies and leading companies within a specific industry.

Source: Tsesmelis, M., Dolamic, L., Keupp, M. M., Percia David, D., & Mermoud, A. (2022). Identifying Emerging Technologies and Leading Companies using Network Dynamics of Patent Clusters: a Cybersecurity Case Study. Arxiv, Article arXiv:2209.10224. 

------
## Abstract
Strategic decisions rely heavily on non-scientific instrumentation to forecast emerging technologies and leading companies. Instead, we build a fast quantitative system with a small computational footprint to discover the most important technologies and companies in a given field, using generalisable methods applicable to any industry. With the help of patent data from the US Patent and Trademark Office, we first assign a value to each patent thanks to automated machine learning tools. We then apply network science to track the interaction and evolution of companies and clusters of patents (i.e. technologies) to create rankings for both sets that highlight important or emerging network nodes thanks to five network centrality indices. Finally, we illustrate our system with a case study based on the cybersecurity industry. Our results produce useful insights, for instance by highlighting (i) emerging technologies with a growing mean patent value and cluster size, (ii) the most influential companies in the field and (iii) attractive startups with few but impactful patents. Complementary analysis also provides evidence of decreasing marginal returns of research and development in larger companies in the cybersecurity industry.

------
## Documents
For more information please refer to [our paper](https://arxiv.org/abs/2209.10224). Mathematical formulas are substantiated and the system is applied to a use case in cybersecurity technology monitoring.

------
## System description

**Data**:
[Patentsview](https://www.patentsview.org).
The data is freely available from the bulk data download page (https://patentsview.org/download/data-download-tables)

**Methodology** :
Our system works with three sets of entities: "Patents", "Assignees" and "CPC subgroups".

Each patent is usually sponsored by one or more assignees, which are organisations or individuals that have an ownership interest in the patent claims.
CPC is a classification scheme which clusters patents according to the subject matter of the patented innovation. CPC groups innovations into 242,050 predefined technological clusters.

Our recommender system consists of four layers, shown in the flowchart below. First, patent data is cleaned in the Data Preprocessing layer. Secondly, in the Machine Learning layer, we cluster these patents and extract key descriptive features to
train machine learning classifiers. The output of the Machine Learning layer is a labeled set of patents, split between low- and high-value patents. A user inputs keywords into the Managerial layer tied to a specific area of interest or industry. Based on these keywords, the Managerial layer selects relevant patents thanks to keyword extraction and passes
these patents further on. The Network Analytics layer receives these topical patents and builds a graph of patent clusters (i.e. CPC subgroups) and patent assignees (i.e.
companies) associated to these topical patents. Finally, our system uses this graph to build useful indicators and rankings of both sets of graph nodes.

![Flowchart](plots/flowchart.png)

**Indicators** :
Five indicators describe dominant nodes in the graph according to different criteria.

Technology Index: calculates the emergingness of a technology based on CPC subgroup patent count growth and mean patent value growth.
Assignee Quality Index: ranks companies according to the quality of their patents.
Impact Index: measures the value of the contribution of an assignee to a query and rewards assignees that have strong connections to important technologies in the network.
Normalised Impact Index: proportionally weighted version of the Impact Index. Some smaller companies with a high proportion of high-value patents will not appear prominently in the Impact Index due to their small number of patents. 
Influence Index: uses eigenvector centrality to highlight well-connected and influential companies in the network.


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
| uspatentcitation.tsv | Citations made to US granted patents by US patents | uuid, patent_id, citation_id, date, name, kind, country, category, sequence | 08.10.2021 |

------
## How to run
Place all Patentsview tsv files into the ``data/patentsview_data`` folder.

Load these datasets into lighter dictionaries:
```
python3 tensor_deployment.py
```

Change job configurations under ``functions/config.py``, under ``class MlConfig``:
- number of CPU cores (self.number_of_cores)
- machine learning maximum search time (self.ml_search_time). 6-12 hours is 
- size of dataframe fed to the machine learning classifiers (self.size_dataframe_train). 15'000 is a good number.
- graph name (self.graph_name)
- keywords specifying the industry (self.keyphrases). Several keyword lists can be given, and each search runs on the same labeled set of patents outputed by the machine learning phase.

Run Managerial layer:
```
python3 main.py
```

The Managerial layer allows you to:
1) create a new graph (required when using this library the first time). 
2) inspect what graph nodes are linked to a specific technology or company. 
3) display abstracts of topical patents to tweak the keyword selection.
4) create rankings based on the five indicators mentioned above

To generate plots depicting the statistical distribution of the data among other graphs:
```
python3 plots.py
```