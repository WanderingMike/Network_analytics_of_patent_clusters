from functions.functions_plots import *

def trivariate_heatmap():
    '''
    Draw a heatmap with patent mainclass as y value, year as x value, and colour of each cell tied to the amount of
    patents.
    '''

    df = pd.read_pickle("data/dataframes/filled_df.pkl")
    df = df[["date", "forward_citations", "MF"]]
    print(df)
    df["year"] = df["date"].apply(lambda x: x.year)
    df = df.explode('MF')
    df = df[df["MF"].notna()]
    print(df)
    df = df[df["forward_citations"].notna()]
    df["MF"] = df["MF"].apply(lambda x: x[0])
    print(df)
    df.drop_duplicates(inplace=True)
    print(df)

    final_df = df.groupby(["MF", "year"], as_index=False).mean()
    final_df = final_df.pivot(index="MF", columns='year', values='forward_citations')

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(final_df, linewidths=.5, ax=ax)
    plt.show()
    plt.savefig("plots/heatmap_year_vs_mainclass_vs_citation_count.png", bbox_inches='tight')
    plt.close()


def cdf_plot(data_series, indicator):
    '''Builds a cumulative distribution function
    :param data_series: main dataset
    :param indicator: independent variable of the cdf
    '''

    data = [point[indicator] for point in data_series]

    x_axis = np.sort(data)
    n = len(data)
    y_axis = np.array(range(n)) / float(n)

    plt.plot(x_axis, y_axis)
    plt.show()
    plt.savefig("plots/cdf_{}.png".format(indicator), bbox_inches='tight')
    plt.close()


def violin_graph(data, indicator, ylim):
    '''Draws a violin graph time-series
    :param data: data distribution
    :param ylim: plot y axis limit'''

    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(11, 6))
    ax.set(ylim=ylim)

    # Draw a nested violinplot and split the violins for easier comparison
    sns.violinplot(data=data, x="year", y="value", hue="type",
                   split=True, inner="quart", linewidth=1,
                   palette={"query": "b", "complete": ".85"})
    sns.despine(left=True)

    plt.show()
    plt.savefig("plots/violin_plot_{}_query_complete.png".format(indicator), bbox_inches='tight')
    plt.close()


def violin_plots_time_series():
    '''Builds a time-series of violin plots for patent count and patent value'''

    cpc_time_series = load_pickle("data/clusters.pkl")
    topical_clusters = pd.read_csv("output_tables/clusters_df.csv")["CPC"].tolist()
    topical_clusters_time_series = {group: cpc_time_series[group] for group in topical_clusters}

    df_emergingness = prepare_violin_plot_df(cpc_time_series, topical_clusters_time_series, "emergingness")
    df_patent_count = prepare_violin_plot_df(cpc_time_series, topical_clusters_time_series, "patent_count")

    violin_graph(df_emergingness, "emergingness", (0,1))
    violin_graph(df_patent_count, "patent_count", (0, np.percentile(df_patent_count["value"], 95)))


def network_draw_subgraph():

    network = load_pickle("data/plots/network.pkl")
    most_connected = list(network.degree())
    most_connected_clusters = [node[0] for node in most_connected if "/" in node[0]][:10]
    most_connected_assignees = [node[0] for node in most_connected if "/" not in node[0]][:10]
    most_connected = most_connected_assignees + most_connected_clusters

    subnetwork = network.subgraph(most_connected)
    print(subnetwork.edges())
    nx.draw(subnetwork)
    plt.show()
    nx.draw_networkx_labels(subnetwork, pos=nx.spring_layout(network))
    nx.draw_networkx_edges(subnetwork, pos=nx.circular_layout(network))
    plt.show()


def scatterplot(df, x, y, xlim, ylim, focus):
    '''
    Builds a scatterplot graph
    :param df: dataset
    :param x: dependent variable
    :param y: independent variable
    :param xlim: ticker limits on x-axis
    :param ylim: ticker limits on y-axis
    :param focus: zoomed or unzoomed version of graph
    '''

    sns.set_theme(style="white", color_codes=True)

    # Use JointGrid directly to draw a custom plot
    g = sns.JointGrid(data=df, x=x, y=y, space=0, ratio=17, xlim=xlim, ylim=ylim)
    g.plot_joint(sns.scatterplot, size=df["count"], sizes=(30, 120),
                 color="g", alpha=.6, legend=False)
    g.plot_marginals(sns.rugplot, height=1, color="g", alpha=.6)

    plt.show()
    plt.savefig("plots/scatterplot_{}_vs_{}_{}.png".format(x, y, focus), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    ### Plot 1: Trivariate heatmap
    trivariate_heatmap()

    ### Plot 2 & 3: cumulative distribution functions
    # emergingness_data = cdf_data()
    # save_pickle("data/plots/emergingness_data.pkl", data=emergingness_data)
    emergingness_data = load_pickle("data/plots/emergingness_data.pkl")
    cdf_plot(emergingness_data, indicator="emergingness")
    cdf_plot(emergingness_data, indicator="patent_count")

    ### Plot 4 & 5: violin plots
    violin_plots_time_series()

    ### Plot 6: network subgraph
    network_draw_subgraph()

    ### Plot 7 & 8: scatterplots
    df = pd.read_csv("output_tables/assignee_df.csv")
    scatterplot(df, "count", "normalised impact",
                xlim=(0, df["count"].max()),
                ylim=(0, df["normalised impact"].max()),
                focus="unzoomed")
    scatterplot(df, "count", "normalised impact",
                xlim=(0, np.percentile(df["count"], 99)),
                ylim=(0, np.percentile(df["normalised impact"], 99)),
                focus="zoomed")
    scatterplot(df, "influence", "impact",
                xlim=(0, df["influence"].max()),
                ylim=(0, df["impact"].max()),
                focus="unzoomed")
    scatterplot(df, "influence", "impact",
                xlim=(0, np.percentile(df["influence"], 99.5)),
                ylim=(0, np.percentile(df["impact"], 99.5)),
                focus="zoomed")

