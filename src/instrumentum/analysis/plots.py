# Individual functions to plot advanced analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe
import logging
import matplotlib.ticker as mtick


from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from optbinning import OptimalBinning

# TODO: poner la categoria missing al final!
# TODO: el color del mean no respeta el orden!
def plot_categorical_with_binary_target(
    df,
    category,
    target,
    ax,
    target_true=1,
    cluster=None,
    copy_dataframe=True,
    verbose=logging.WARNING,
    palette="husl",
    sort_func=None,
):

    COL_GRP_ALL = "all"
    COL_GRP_TARGET = "target"

    MISSING_VALUE = "Missing"

    HATCH = "////\\\\\\\\"

    colors = sns.color_palette(palette)

    if cluster and df[cluster].nunique() == 1:
        cluster = None
        logging.warning("Only one category in cluster. Clustering visual removed")

    cols_received = [category, target] + ([cluster] if cluster is not None else [])
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
            df[df[target] == target_true].groupby(cols_to_groupby, sort=False).size(),
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

    # Plot a bar in front of the previous one, just to highlight the % of events in that category
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
        **kwargs
    )

    ax_r = ax.twinx()
    data = (grp[COL_GRP_TARGET] / grp[COL_GRP_ALL]).reset_index(name="percentage")

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
            path_effects=[pe.Stroke(linewidth=2, foreground="black"), pe.Normal()],
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
    # (2) Remove the right side default handles (if no cluster, it doesnt exist. That's the check)
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
    cut = pd.cut(df[~mask_n][x], bins=bins, labels=splits_str, include_lowest=True)

    df["category"] = np.nan
    df.loc[~mask_n, "category"] = cut.astype(str)

    df.sort_values(by=["category"], inplace=True)

    plot_categorical_with_binary_target(
        df, category="category", target=y, cluster=cluster, ax=ax, palette=palette
    )


if __name__ == "__main__":

    if False:
        df = pd.DataFrame(
            {
                "cluster": ["A", "A", "B", "B", "A", "B", "C", "C", "C", "A", "A", "A"],
                "category": [
                    "x",
                    "y",
                    "x",
                    "x",
                    "x",
                    np.nan,
                    "y",
                    "z",
                    np.nan,
                    np.nan,
                    "z",
                    "z",
                ],
                "result": [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
            }
        )

        f, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))

        plot_categorical_with_binary_target(
            df, cluster="cluster", category="category", target="result", ax=ax1
        )

        plot_categorical_with_binary_target(
            df[df["cluster"] == "A"], category="category", target="result", ax=ax2
        )

        plt.show()

    if True:

        np.random.seed(0)

        df1 = pd.DataFrame()
        df1["dist"] = np.random.normal(0, 20, size=1000)
        df1["clust"] = "A"
        df1["target"] = 0
        df1_ = pd.DataFrame()
        df1_["dist"] = np.random.normal(3, 30, size=200)
        df1_["clust"] = "A"
        df1_["target"] = 1

        df2 = pd.DataFrame()
        df2["dist"] = np.random.normal(3, 35, size=1000)
        df2["clust"] = "B"
        df2["target"] = 0
        df2_ = pd.DataFrame()
        df2_["dist"] = np.random.normal(2, 20, size=200)
        df2_["clust"] = "B"
        df2_["target"] = 1

        df3 = pd.DataFrame()
        df3["dist"] = np.random.normal(50, 90, size=1000)
        df3["clust"] = "C"
        df3["target"] = 0
        df3_ = pd.DataFrame()
        df3_["dist"] = np.random.normal(55, 95, size=200)
        df3_["clust"] = "C"
        df3_["target"] = 1

        df = pd.concat([df1, df1_, df2, df2_, df3, df3_], ignore_index=True)

        df.loc[3, "dist"] = np.nan
        f, ax = plt.subplots(figsize=(15, 7))

        plot_continuos_bin_with_binary_target(
            df, x="dist", cluster="clust", master_clust="A", y="target", ax=ax
        )
        plt.show()
