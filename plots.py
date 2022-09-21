import matplotlib.pyplot as plt

from functions.functions_plots import *
import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22
hfont = {'fontname': 'Times New Roman'}
font_size = 18
sns.set(font="Times New Roman")


def trivariate_heatmap(load=False):
    """
    Draw a heatmap with patent mainclass as y value, year as x value, and colour of each cell tied to the amount of
    patents.
    """

    def get_mainclass(mainclasses):
        answer = list()
        for mainclass in mainclasses:
            answer.append(mainclass[0])
        return list(set(answer))

    if load:
        final_df = load_pickle("data/plots/heatmap_df.pkl")
    else:
        df = pd.read_pickle("data/dataframes/filled_df.pkl")
        df = df[["date", "forward_citations", "MF"]]
        print(df)
        df["Mainclass"] = df["MF"].apply(get_mainclass)
        print(df)
        df["year"] = df["date"].apply(lambda x: x.year)
        df = df.explode('Mainclass')
        df = df[df["Mainclass"].notna()]
        df = df[df["forward_citations"].notna()]
        print(df)

        final_df = df.groupby(["Mainclass", "year"], as_index=False).mean()
        final_df = final_df.pivot(index="Mainclass", columns='year', values='forward_citations')
        save_pickle("data/plots/heatmap_df.pkl", final_df)

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(final_df, cmap="Greys", center=3, linewidths=.5, ax=ax)
    plt.subplots_adjust(top=0.9)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    #fig_title = "Average forward citation count per mainclass category per year"
    #plt.suptitle(fig_title, fontsize=font_size)
    plt.xticks(fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    plt.xlabel("year", fontsize=22)
    plt.ylabel("section", fontsize=22)
    plt.locator_params(axis='x', nbins=16)
    plt.savefig("plots/{0}/heatmap_year_vs_mainclass_vs_citation_count.{0}".format(image_format), dpi=400)
    plt.show()
    plt.close()


def cdf_plot(data_series, indicator, fig_title):
    """
    Builds a cumulative distribution function for all topical clusters.
    :param data_series: main dataset with the value of each cluster and its patent count
    :param indicator: independent variable of the cdf
    :param fig_title: title of figure
    """

    data = [point[indicator] for point in data_series]

    x_axis = np.sort(data)
    n = len(data)
    y_axis = np.array(range(n)) / float(n)

    plt.plot(x_axis, y_axis)
    plt.subplots_adjust(top=0.9)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.suptitle("Cumulative Distribution Function for {}".format(fig_title), fontsize=font_size)
    plt.legend(labels=["CDF"], fontsize=22, loc="lower right")
    plt.xlabel(fig_title, fontsize=22)
    plt.ylabel("share of clusters", fontsize=22)
    plot_name = "plots/{0}/cdf_{1}.{0}".format(image_format, indicator)
    plt.savefig(plot_name, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()


def violin_graph(data, indicator, ylim, fig_title):
    """Draws a violin graph time-series
    :param indicator: graph file name
    :param fig_title: figure title
    :param data: data distribution
    :param ylim: plot y axis limit
    """

    sns.set_theme(style="whitegrid", font="Times New Roman")
    f, ax = plt.subplots(figsize=(11, 6))
    ax.set(ylim=ylim)

    # Draw a nested violinplot and split the violins for easier comparison
    sns.violinplot(data=data, x="year", y="cluster value", hue="type",
                   split=True, inner="quart", linewidth=1,
                   palette={"full": "b", "query": ".85"})

    sns.despine(left=True)

    plt.subplots_adjust(top=0.9)
    #plt.suptitle("Violin plots depicting the probability density for {}".format(fig_title),
    #             fontsize=font_size+4,
    #             **hfont)
    plot_name = "plots/{0}/violin_plot_{1}_query_full.{0}".format(image_format, indicator)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("year", fontsize=22)
    plt.ylabel("cluster value", fontsize=22)
    ax.legend(bbox_to_anchor=(1.14, 1.18), title="selected patents", title_fontsize=20, fontsize=16)
    plt.savefig(plot_name, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()


def violin_plots_time_series(cpc_time_series, topical_clusters):
    """Builds a time-series of violin plots for patent count and patent value"""

    topical_clusters_time_series = {group: cpc_time_series[group] for group in topical_clusters}

    df_emergingness = prepare_violin_plot_df(cpc_time_series, topical_clusters_time_series, "emergingness")
    df_patent_count = prepare_violin_plot_df(cpc_time_series, topical_clusters_time_series, "patent_count")

    violin_graph(df_emergingness, "emergingness", (0, 1), fig_title="cluster value")
    violin_graph(df_patent_count,
                 "patent_count",
                 (0, np.percentile(df_patent_count["cluster value"], 99)),
                 fig_title="cluster size")


def network_draw_subgraph():

    network = load_pickle("data/network.pkl")
    most_connected = list(network.degree())
    most_connected_clusters = [node[0] for node in most_connected if "/" in node[0]][:10]
    most_connected_assignees = [node[0] for node in most_connected if "/" not in node[0]][:10]
    most_connected = most_connected_assignees + most_connected_clusters

    subnetwork = network.subgraph(most_connected)
    print(subnetwork.edges())
    nx.draw(subnetwork)
    # plt.show()
    nx.draw_networkx_labels(subnetwork, pos=nx.spring_layout(network))
    nx.draw_networkx_edges(subnetwork, pos=nx.circular_layout(network))
    # plt.show()


def scatterplot(df, x, y, xlim, ylim, x_title, y_title, focus):
    """
    Builds a scatterplot graph
    :param df: dataset
    :param x: dependent variable
    :param y: independent variable
    :param xlim: ticker limits on x-axis
    :param ylim: ticker limits on y-axis
    :param focus: zoomed or unzoomed version of graph
    """

    sns.set_theme(style="white", color_codes=True, font="Times New Roman")
    # Use JointGrid directly to draw a custom plot
    g = sns.JointGrid(data=df, x=x, y=y, space=0, ratio=17, xlim=xlim, ylim=ylim)
    g.set_axis_labels(x_title, y_title, fontsize=16)
    g.plot_joint(sns.scatterplot, size=df["count"], sizes=(30, 120),
                 color="g", alpha=.6, legend=False)

    plt.subplots_adjust(top=0.9)

    #plt.suptitle("{} vs {}".format(x_title, y_title), fontsize=font_size, **hfont)
    plot_name = "plots/{0}/scatterplot_{1}_vs_{2}_{3}.{0}".format(image_format, x, y, focus)
    plt.savefig(plot_name, bbox_inches='tight', dpi=400)
    plt.show()
    plt.close()


def interaction(numbers):
    """Runs appropriate functions based on user request"""

    if len({2, 3}.intersection(numbers)) != 0:
        clusters_df = pd.read_csv("output_tables/clusters_df.csv")
        topical_clusters = clusters_df["subgroup"].tolist()
        cpc_time_series = load_pickle("data/ultimate/clusters.pkl")

    for number in numbers:

        print("Plot {}".format(number))

        if number == 1:
            ### Plot 1: Trivariate heatmap
            trivariate_heatmap(load=True)

        elif number == 2:
            ### Plot 2 & 3: cumulative distribution functions
            emergingness_data = cdf_data(cpc_time_series, topical_clusters)
            save_pickle("data/plots/emergingness_data.pkl", data=emergingness_data)
            # emergingness_data = load_pickle("data/plots/emergingness_data.pkl")
            cdf_plot(emergingness_data, indicator="emergingness", fig_title="cluster value")
            cdf_plot(emergingness_data, indicator="patent_count", fig_title="cluster size")

        elif number == 3:
            ### Plot 4 & 5: violin plots
            violin_plots_time_series(cpc_time_series, topical_clusters)

        elif number == 4:
            ### Plot 6: network subgraph
            network_draw_subgraph()

        elif number == 5:
            ### Plot 7 & 8: scatterplots
            assignee_df = pd.read_csv("output_tables/assignee_df.csv")
            scatterplot(assignee_df, "count", "normalised impact",
                        xlim=(0, assignee_df["count"].max()),
                        ylim=(0, assignee_df["normalised impact"].max()),
                        x_title="Assignee patent count",
                        y_title="Normalised impact",
                        focus="unzoomed")
            scatterplot(assignee_df, "count", "normalised impact",
                        xlim=(0, np.percentile(assignee_df["count"], 99)),
                        ylim=(0, np.percentile(assignee_df["normalised impact"], 99)),
                        x_title="Assignee patent count",
                        y_title="Normalised impact",
                        focus="zoomed")
            scatterplot(assignee_df, "influence", "impact",
                        xlim=(0, assignee_df["influence"].max()),
                        ylim=(0, assignee_df["impact"].max()),
                        x_title="Assignee influence",
                        y_title="Assignee impact",
                        focus="unzoomed")
            scatterplot(assignee_df, "influence", "impact",
                        xlim=(0, np.percentile(assignee_df["influence"], 99.5)),
                        ylim=(0, np.percentile(assignee_df["impact"], 99.5)),
                        x_title="Assignee influence",
                        y_title="Assignee impact",
                        focus="zoomed")

        print("Finished plot {}".format(number))


if __name__ == "__main__":
    plot_numbers = input("Which plots would you like to create?\n"
                         "1) Trivariate heatmap\n"
                         "2) Cumulative distribution functions\n"
                         "3) Violin plots\n"
                         "4) Network subgraph\n"
                         "5) Scatterplots\n")
    image_format = input("Image format (pdf, png, jpeg,...): ")
    interaction([int(num) for num in plot_numbers.split()])

