# Individual functions to plot advanced analysis
import logging

import matplotlib.patheffects as pe
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from optbinning import OptimalBinning


# TODO: poner la categoria missing al final!
# TODO: el color del mean no respeta el orden!
def plot_categorical_with_binary_target(
    df,
    category,
    y,
    ax,
    target_true=1,
    cluster=None,
    copy_dataframe=True,
    verbose=logging.WARNING,
    palette="husl",
    sort_func=None,
):

    COL_GRP_ALL = "all"
    COL_GRP_TARGET = "y"

    MISSING_VALUE = "Missing"

    HATCH = "////\\\\\\\\"

    colors = sns.color_palette(palette)

    if cluster and df[cluster].nunique() == 1:
        cluster = None
        logging.warning(
            "Only one category in cluster. Clustering visual removed"
        )

    cols_received = [category, y] + ([cluster] if cluster is not None else [])
    cols_to_groupby = ([cluster] if cluster else []) + [category]

    # Some Validations
    if not all(x in df.columns for x in cols_received):
        raise ValueError("Column names must exist in dataframe")

    # Let's get started
    df = df[cols_received].copy() if copy_dataframe else df[cols_received]

    df[category].fillna(value=MISSING_VALUE, inplace=True)

    # TODO: agregar que no existan nans en otras columnas.

    # General dataframe with groupings, this will be used later by all plots
    grp = pd.concat(
        [
            df.groupby(cols_to_groupby, sort=False).size(),
            df[df[y] == target_true]
            .groupby(cols_to_groupby, sort=False)
            .size(),
        ],
        axis=1,
    ).fillna(0)

    # let's give the columns a name for easier reference
    grp.columns = [COL_GRP_ALL, COL_GRP_TARGET]

    if sort_func:
        sort_func(grp)

    # Plot the a bar with the percentage of each category
    data = (
        grp[COL_GRP_ALL]
        / (
            grp[COL_GRP_ALL].groupby(level=0).sum()
            if cluster
            else grp[COL_GRP_ALL].sum()
        )
    ).reset_index(name="percentage")

    sns.barplot(
        data=data,
        x=category,
        y="percentage",
        hue=cluster,
        dodge=True,
        ax=ax,
        palette=palette,
    )

    handle_clusters, _ = ax.get_legend_handles_labels()

    # Plot a bar in front of the previous one, just to highlight the % of
    # events in that category
    # (if stacked functionality worked I woudnt need to do this crap)
    kwargs = {
        "linewidth": 0,
        "edgecolor": "k",
        "facecolor": "none",
        "hatch": [HATCH],
        "alpha": 0.5,
    }
    data = (
        grp[COL_GRP_TARGET]
        / (
            grp[COL_GRP_ALL].groupby(level=0).sum()
            if cluster
            else grp[COL_GRP_ALL].sum()
        )
    ).reset_index(name="percentage")

    sns.barplot(
        data=data,
        ci=None,
        x=category,
        y="percentage",
        hue=cluster,
        dodge=True,
        ax=ax,
        palette=palette,
        **kwargs,
    )

    ax_r = ax.twinx()
    data = (grp[COL_GRP_TARGET] / grp[COL_GRP_ALL]).reset_index(
        name="percentage"
    )

    sns.lineplot(
        data=data,
        x=category,
        y="percentage",
        hue=cluster,
        linestyle="solid",
        palette=palette,
        marker="o",
        ax=ax_r,
        path_effects=[pe.Stroke(linewidth=2, foreground="black"), pe.Normal()],
    )

    data = (
        grp[COL_GRP_TARGET].groupby(level=[0]).sum()
        / grp[COL_GRP_ALL].groupby(level=[0]).sum()
        if cluster
        else [grp[COL_GRP_TARGET].sum() / grp[COL_GRP_ALL].sum()]
    )

    for e, mean in enumerate(data):
        ax_r.axhline(
            mean,
            ls=":",
            color=colors[e],
            path_effects=[
                pe.Stroke(linewidth=2, foreground="black"),
                pe.Normal(),
            ],
        )

    # Handles and Legends processing
    # (1) Create custom elements
    legend_elements = [
        Patch(
            facecolor="none",
            linewidth=0,
            edgecolor="k",
            hatch=HATCH,
            label="% of events",
        ),
        Line2D([0], [0], color="k", label="% of events within bin."),
        Line2D([0], [0], color="k", ls=":", label="Mean of % events in frame"),
    ]
    # (2) Remove the right side default handles (if no cluster, it doesnt
    # exist. That's the check)
    if ax_r.get_legend():
        ax_r.get_legend().remove()

    # (3) Create the list of handles to be used - If there is only one cluster
    # there is no need of adding it as a legend (redundant)

    handles = (handle_clusters if cluster else []) + legend_elements

    ax.legend(handles=handles, loc="upper right")

    ax.tick_params(axis="x", labelrotation=60)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax_r.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    ax.set(
        ylabel="Percentage of observations by bin",
        title="Percentages by bin",
    )
    ax_r.set(ylabel="Percentage of events in each bin")

    return ax


# TODO: add validations for master cluster
def plot_continuos_bin_with_binary_target(
    df,
    x,
    y,
    ax,
    target_true=1,
    cluster=None,
    master_clust=None,
    verbose=logging.WARNING,
    palette="husl",
):

    df = df.copy()

    # First part is to get the bins, we are using an external function for that
    master_df = df[df[cluster] == master_clust] if cluster else df
    mask_n = master_df[x].isna()

    optb = OptimalBinning(name=x, dtype="numerical", solver="cp")
    optb.fit(master_df[~mask_n][x].values, master_df[~mask_n][y])
    bins = optb.splits

    bins = np.append(bins, np.inf)
    bins = np.insert(bins, 0, -np.inf)

    logging.debug("Bins generated: " + str(bins))

    splits_str = []

    # We add the number at the beginning so it can be sorted
    for count in range(len(bins) - 1):
        beg = "(" + str(count) + ") " + str(round(bins[count], 2))
        end = str(round(bins[count + 1], 2))
        splits_str.append(beg + " to " + end)

    mask_n = df[x].isna()
    cut = pd.cut(
        df[~mask_n][x], bins=bins, labels=splits_str, include_lowest=True
    )

    df["category"] = np.nan
    df.loc[~mask_n, "category"] = cut.astype(str)

    df.sort_values(by=["category"], inplace=True)

    plot_categorical_with_binary_target(
        df, category="category", y=y, cluster=cluster, ax=ax, palette=palette
    )


def plot_value_distribution(
    df,
    ax,
    cluster=None,
    y=None,
    palette="husl",
    call_back_pre_process=None,
    remove_non_nan_cols=True,
    values=[np.nan, pd.NA],
):

    # These columns will be created and plotted:

    # Without considering clusters, the average of % nan by variable
    FULL_AVG_NAN = "FULL_AVG_NAN"

    # The standard deviation there exists within the
    # clusters of each column regarding % nan
    # we can discover if there are differences among the clusters for same
    # columns
    CLUSTER_SD_NAN = "CLUSTER_SD_NAN"

    # The average of the % nan for the clusters of each column
    CLUSTER_AVG_NAN = "CLUSTER_AVG_NAN"

    # If there are a y and no cluster, the SD of the % nan for each y value
    # we can discover if there are differences in the amount of nan for
    # the same column but different cluster
    TARGET_SD_NAN = "TARGET_SD_NAN"

    # If there are y and cluster, the average of TARGET_SD_NAN within each
    # cluster
    TARGET_SD_CLUSTER_AVG_NAN = "TARGET_SD_CLUSTER_AVG_NAN"

    df = df.copy()

    cols_to_nan = [x for x in df.columns if x not in [cluster, y]]
    df[cols_to_nan] = df[cols_to_nan].isin(values) * 1

    # Average by column, without considering clusters (will be used as the
    # master)
    final_df = df[cols_to_nan].T.mean(axis=1).rename(FULL_AVG_NAN)

    if remove_non_nan_cols:
        final_df = final_df[final_df > 0]

    if cluster:

        gby = df.groupby(cluster)[cols_to_nan].mean().T

        gby[CLUSTER_SD_NAN] = gby.std(axis=1)
        gby[CLUSTER_AVG_NAN] = gby.mean(axis=1)

        final_df = pd.merge(final_df, gby, left_index=True, right_index=True)

        if y:
            # (1) Calculate the mean of nans by cluster and y
            # (2) Calculate the STD by y (within each cluster)
            # (3) Calculate the AVG of those STD by cluster
            gby = (
                df.groupby([cluster, y])
                .mean()
                .groupby(level=0)
                .std()
                .T.mean(axis=1)
                .rename(TARGET_SD_CLUSTER_AVG_NAN)
            )
            final_df = pd.merge(
                final_df, gby, left_index=True, right_index=True
            )

    if y:
        gby = df.groupby(y).mean().T.std(axis=1).rename(TARGET_SD_NAN)
        final_df = pd.merge(final_df, gby, left_index=True, right_index=True)

    if call_back_pre_process:
        final_df = call_back_pre_process(final_df, cluster, y)

    if cluster:
        melted = final_df.reset_index().melt(
            id_vars="index", value_vars=df[cluster].unique()
        )
        x_val, y_val, hue_val = (
            melted["index"],
            melted["value"],
            melted["variable"],
        )
    else:
        x_val, y_val, hue_val = (
            final_df.index,
            final_df[FULL_AVG_NAN] if y else final_df.values,
            None,
        )

    sns.barplot(x=x_val, y=y_val, hue=hue_val, ax=ax, palette=palette)
    ax.tick_params(axis="x", labelrotation=90)
    ax.set(
        ylabel="Percentage of nan",
        xlabel=None,
        title="Nan Analysys",
    )

    if not y and not cluster:
        return ax

    # Lines ------------

    # I want to make sure that I use exactly the same xticks and in the same
    # order as that placed in the barplot (I could assume that since I used the
    # same dataframe) but just in case
    df_lines = pd.DataFrame(
        [tick.get_text() for tick in ax.get_xticklabels()], columns=["col"]
    )
    df_lines = df_lines.join(final_df, on="col").fillna(0)

    handle_bar, _ = ax.get_legend_handles_labels()
    handles = []
    colors = sns.color_palette(palette)

    ax_r = ax.twinx()

    # at least one have to be due to the previous if return
    lines_to_print = (
        ["CLUSTER_SD_NAN", "TARGET_SD_CLUSTER_AVG_NAN"]
        if cluster and y
        else (["TARGET_SD_NAN"] if y else ["CLUSTER_SD_NAN"])
    )

    for count, line in enumerate(lines_to_print):
        sns.lineplot(
            data=df_lines,
            x="col",
            y=line,
            linestyle="solid",
            marker="o",
            ax=ax_r,
            color=colors[-count],
            path_effects=[
                pe.Stroke(linewidth=2, foreground="black"),
                pe.Normal(),
            ],
        )

        handles += [Line2D([0], [0], color=colors[-count], label=line)]

    handles = handle_bar + handles
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.4, 1))

    ax_r.set(ylabel="Standard Deviations")

    return ax_r


def plot_value_heatmap(
    df,
    values=[np.nan, pd.NA],
):
    df = df.copy()
    df = df.isin(values) * 1

    bins_size = 100
    n_bins = int(len(df) / bins_size)

    n = df.groupby(pd.cut(df.index, n_bins)).sum()

    n = n / bins_size
    n.clip(upper=1, inplace=True)

    g = sns.clustermap(
        n,
        cmap="RdYlGn_r",
        col_cluster=True,
        row_cluster=False,
        # yticklabels=False,
    )  # cbar_kws={'label': 'Missing Data'})

    g.ax_col_dendrogram.set_title("Heatmap of value's occurrence")
    # for a in g.ax_col_dendrogram.collections:
    #     a.set_linewidth(1)
    #     a.set_color("grey")
