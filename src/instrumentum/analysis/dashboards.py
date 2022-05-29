# Individual functions to plot advanced analysis
import logging

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from instrumentum.analysis.plots import (
    plot_categorical_with_binary_target,
    plot_continuos_bin_with_binary_target,
)


def dashboard_continuos_with_binary_target(
    df,
    x,
    y,
    palette="husl",
    target_true=1,
    cluster=None,
    copy_dataframe=True,
    verbose=logging.WARNING,
):

    if df[y].nunique() > 2:
        raise ValueError("This dashboard shows only binary targets")

    if target_true not in df[y].unique():
        raise ValueError("Parameter target_true is not an actual value of y")

    if copy_dataframe:
        df = df.copy()

    if cluster:
        df[cluster] = df[cluster].astype("str")

    ncols = 3
    nrows = 3
    fig, axs = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(6 * ncols, 5 * nrows)
    )

    # sns.barplot(
    #     data=df,
    #     x=cluster if cluster else y,
    #     y=x if cluster else x,
    #     hue=y if cluster else None,
    #     ax=axs[0, 0],
    #     palette=palette,
    #     ci=None,
    #     estimator=len,
    # )
    sns.countplot(
        data=df,
        x=y,
        hue=cluster,
        ax=axs[0, 0],
        palette=palette,
    )

    axs[0, 0].set_title("Elements Count by Target")

    sns.boxplot(
        x=cluster if cluster else y,
        y=x,
        data=df,
        hue=y if cluster else None,
        ax=axs[0, 1],
        palette=palette,
    )
    axs[0, 1].set_title("Box Plot")

    sns.stripplot(
        x=cluster if cluster else y,
        y=x,
        data=df,
        hue=y if cluster else None,
        ax=axs[0, 2],
        s=1,
        dodge=True,
        jitter=True,
        palette=palette,
    )
    axs[0, 2].set_title("Strip Plot")

    ax_large = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax_large.grid()

    ##

    extra_legend_elements = []
    styles = {True: "solid", False: "dotted"}

    for c, t in enumerate(df[y].unique()):
        sns.kdeplot(
            x=x,
            data=df[df[y] == t],
            ax=ax_large,
            linewidth=2,
            linestyle=styles[t == target_true],
            fill=t == target_true,
            hue=cluster,
            common_norm=False,
            palette=palette,
            # legend=False,
            path_effects=[
                pe.Stroke(linewidth=2, foreground="black"),
                pe.Normal(),
            ],
        )

        extra_legend_elements += [
            Line2D(
                [0],
                [0],
                color="k",
                ls=styles[t == target_true],
                label="Target equal to " + str(t),
            )
        ]

    handles = ax_large.legend_.legendHandles if ax_large.legend_ else []

    if handles:
        for h, t in zip(handles, ax_large.legend_.texts):
            h.set_label(
                t.get_text()
            )  # assign the legend labels to the handles

    ax_large.legend(handles=handles + extra_legend_elements, loc="upper right")
    ax_large.set_title("Distribution by target")
    # ECDPLOT

    axs[1, 2].grid()
    extra_legend_elements = []
    styles = {True: "solid", False: "dotted"}
    for c, t in enumerate(df[y].unique()):
        sns.ecdfplot(
            x=x,
            hue=cluster,
            data=df[df[y] == t],
            ax=axs[1, 2],
            linewidth=2,
            linestyle=styles[t == target_true],
            palette=palette,
        )
        extra_legend_elements += [
            Line2D(
                [0],
                [0],
                color="k",
                ls=styles[t == target_true],
                label="Target equal to " + str(t),
            )
        ]

    handles = axs[1, 2].legend_.legendHandles if axs[1, 2].legend_ else []

    if handles:
        for h, t in zip(handles, axs[1, 2].legend_.texts):
            h.set_label(
                t.get_text()
            )  # assign the legend labels to the handles

    axs[1, 2].legend(
        handles=handles + extra_legend_elements, loc="lower right"
    )
    axs[1, 2].set_title("Cumulative Distribution")

    ##

    ax_large_2 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

    master_clust = df[cluster].unique()[0] if cluster else None
    plot_continuos_bin_with_binary_target(
        df,
        x,
        y,
        ax=ax_large_2,
        cluster=cluster,
        master_clust=master_clust,
        target_true=target_true,
        palette=palette,
    )

    # fig.tight_layout()
    fig.suptitle("Analysis of column: " + str(x), fontsize=25)

    return fig


# TODO: add an average line in the pointplot
def dashboard_categorical_with_binary_target(
    df,
    x,
    y,
    palette="husl",
    target_true=1,
    cluster=None,
    copy_dataframe=True,
    verbose=logging.WARNING,
):

    if df[y].nunique() > 2:
        raise ValueError("This dashboard shows only binary targets")

    if target_true not in df[y].unique():
        raise ValueError("Parameter target_true is not an actual value of y")

    cols_received = [x, y] + ([cluster] if cluster is not None else [])

    # Some Validations
    if not all(x in df.columns for x in cols_received):
        raise ValueError("Column names must exist in dataframe")

    # Let's get started
    df = df[cols_received].copy() if copy_dataframe else df[cols_received]
    df[x] = df[x].astype("str")
    if cluster:
        df[cluster] = df[cluster].astype("str")

    df["freq"] = df.groupby(x)[x].transform("count")
    df = df.sort_values("freq", ascending=False).drop("freq", axis=1)

    ncols = 3
    nrows = 2
    fig, axs = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(6 * ncols, 5 * nrows)
    )

    ax_large = plt.subplot2grid((nrows, ncols), (1, 0), colspan=3)

    sns.countplot(
        data=df,
        x=y,
        hue=cluster,
        ax=axs[0, 0],
        palette=palette,
    )
    axs[0, 0].set_title("Elements Count by Target")

    ax = axs[0, 1]
    extra_legend_elements = []
    styles = {True: "solid", False: "dotted"}

    for c, t in enumerate(df[y].unique()):
        sns.ecdfplot(
            x=x,
            hue=cluster,
            data=df[df[y] == t],
            ax=ax,
            linewidth=2,
            linestyle=styles[t == target_true],
            palette=palette,
        )
        extra_legend_elements += [
            Line2D(
                [0],
                [0],
                color="k",
                ls=styles[t == target_true],
                label="Target equal to " + str(t),
            )
        ]

    ax.tick_params(axis="x", labelrotation=60)

    handles = ax.legend_.legendHandles if ax.legend_ else []

    if handles:
        for h, t in zip(handles, ax.legend_.texts):
            h.set_label(
                t.get_text()
            )  # assign the legend labels to the handles

    ax.legend(handles=handles + extra_legend_elements, loc="lower right")
    ax.set_title("Cumulative Distribution")

    sns.pointplot(
        x=x,
        y=y,
        hue=cluster,
        data=df,
        ax=axs[0, 2],
        linewidth=1,
        dodge=True,
        palette=palette,
    )
    axs[0, 2].tick_params(axis="x", labelrotation=60)
    axs[0, 2].set_title("Confidence by Category")

    plot_categorical_with_binary_target(
        df,
        category=x,
        y=y,
        ax=ax_large,
        cluster=cluster,
        palette=palette,
        sort_func=lambda grp: grp.sort_values(
            ["all"], ascending=False, inplace=True
        ),
    )

    # fig.tight_layout()
    fig.suptitle("Analysis of column: " + str(x), fontsize=25)

    return fig
