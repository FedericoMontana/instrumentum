import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from optbinning import OptimalBinning
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import numpy as np


class DistAnalyzer:
    def __init__(self, frames, frames_names=None, target=None, target_true=1):

        self.frames = frames
        self.target = target
        self.target_true = target_true

        cols = [f.columns.tolist() for f in frames]
        self.common_cols = set(cols[0]).intersection(*cols)

        if frames_names:
            self.frame_names = frames_names
        else:
            self.frame_names = ["Dataset " + str(x) for x in range(len(frames))]

        if target and target not in self.common_cols:
            raise ValueError("Target not included in all dataframes")

    @property
    def comm_columns(self):
        return list(self.common_cols)

    def get_bins_info_2(self, col):

        ret = {}
        ret["bins"] = []

        # f = self.frames[0]
        # f = f[~f[col].isna()]
        # cuts, bins = pd.qcut(
        #     f[col], q=8, labels=False, retbins=True, precision=1, duplicates="drop"
        # )

        # cuts_str = pd.qcut(f[col], q=8, precision=1, duplicates="drop")
        # ret["splits"] = cuts_str.unique()

        # # ret["splits"].sort()
        # ret["splits_str"] = [
        #     "missing" if x is np.nan else str(x) for x in ret["splits"]
        # ]
        mask_n = self.frames[0].isna()
        x = self.frames[0][~mask_n][col].values
        y = self.frames[0][~mask_n][self.target]

        optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
        optb.fit(x, y)
        bins = optb.splits

        bins = np.append(bins, np.inf)
        bins = np.insert(bins, 0, -np.inf)

        splits_str = []
        for count in range(len(bins) - 1):
            beg = str(round(bins[count], 2))
            end = str(round(bins[count + 1], 2))
            splits_str.append(beg + " to " + end)

        ret["splits_str"] = splits_str
        ret["splits_str"].append("Missing")

        # TODO
        # El bining usando optimalbinning (ver la funcion origninal, no la 2), funcionaba bien
        # pero ocurria que la tabla qeu devolvia si habia un bin con zero lo expandia (ej el original era 23-34, 35-inf y por no tener
        # items luego ponia 23-inf y no quedaban del mismo tamano
        # como esta ahora parece que funciona, pero ojo con el cut de pandas que creo que tambien
        # modifica de acuerdo a la seleccion si no tiene valores
        # ver de usar digitize de numpy

        for e, f in enumerate(self.frames):

            info = {}
            mask_n = f[col].isna()
            mask_event = f[self.target] == self.target_true

            cuts = pd.cut(f[~mask_n][col], bins=bins, labels=False, include_lowest=True)

            # Important, the index of the cuts must be maintained, in case a bin is zero
            # parameter no index will complicate things
            a = f.groupby(cuts).agg({col: "count"})
            b = f[mask_event].groupby(cuts).agg({col: "count"})
            info["event_rate"] = (b / a)[col].values
            info["bin_%"] = (a / len(f))[col].values

            # if a bin is zero, previous division sends a nan
            info["event_rate"][np.isnan(info["event_rate"])] = 0
            info["bin_%"][np.isnan(info["bin_%"])] = 0

            ## Nans, mejorar esto!
            if sum(mask_n) > 0:
                perc_nan = sum(mask_n) / len(f)
                nan_event_rate = sum((mask_n & mask_event) / sum(mask_n))
            else:
                perc_nan = nan_event_rate = 0

            info["event_rate"] = np.append(info["event_rate"], nan_event_rate)
            info["bin_%"] = np.append(info["bin_%"], perc_nan)

            ret["bins"].append(info.copy())

        return ret

    def get_bins_info(self, col):
        ret = {}
        ret["bins"] = []
        for count, f in enumerate(self.frames):
            x = f[col].values
            y = f[self.target]

            info = {}
            if count == 0:
                optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
                optb.fit(x, y)
                bin_table = optb.binning_table.build()
                bin_table = bin_table[~bin_table["Bin"].isin(["Special", ""])]
                ret["splits"] = optb.splits
                ret["splits_str"] = bin_table["Bin"]
            else:
                optb = OptimalBinning(
                    name=col, user_splits=ret["splits"], dtype="numerical", solver="cp"
                )
                optb.fit(x, y)
                bin_table = optb.binning_table.build()

            bin_table = bin_table[~bin_table["Bin"].isin(["Special", ""])]
            info["event_rate"] = bin_table["Event rate"].values
            info["bin_%"] = bin_table["Count (%)"].values

            ret["bins"].append(info.copy())

        return ret

    def get_analysis_frames(self):
        analysis_input = {}
        info = {}
        for n, f in enumerate(self.frames):

            info["# of cols"] = len(f.columns)
            info["# com. cols"] = len(set(f.columns).intersection(self.common_cols))
            info["% com. cols: "] = round(info["# com. cols"] / info["# of cols"], 2)

            if self.target:
                info["# of targets"] = sum(f[self.target] == self.target_true)
                info["% of targets"] = sum(f[self.target] == self.target_true) / len(f)

            analysis_input[self.frame_names[n]] = info

        return analysis_input

    def _plot_individual_distributions(
        self, target, col=None, pal="husl", sd_cutoff=2.5
    ):

        colors = sns.color_palette(pal, len(self.frames))
        dff = pd.DataFrame()

        # Let's create a single dataframe so we can use
        # seaborn and hue
        for con, d in enumerate(self.frames):
            r = d.copy()
            r = r[abs(r[col]) < r[col].mean() + r[col].std() * sd_cutoff]
            r["frame"] = self.frame_names[con]
            dff = dff.append(pd.DataFrame(r), ignore_index=True)

        ncols = 3
        nrows = 2
        fig, axs = plt.subplots(ncols=ncols, nrows=2, figsize=(6 * ncols, 7 * nrows))

        sns.violinplot(
            x="frame",
            y=col,
            data=dff,
            hue=target,
            ax=axs[0, 0],
            split=True,
            palette=pal,
        )
        axs[0, 0].set_title("Violin Plot")

        sns.boxplot(x="frame", y=col, data=dff, hue=target, ax=axs[0, 1], palette=pal)
        axs[0, 1].set_title("Box Plot")

        sns.stripplot(
            x="frame",
            y=col,
            data=dff,
            hue=target,
            ax=axs[0, 2],
            s=1,
            dodge=True,
            jitter=True,
            palette=pal,
        )
        axs[0, 2].set_title("Strip Plot")

        def tick(ax, label, rotation=0):
            ax.set_title(label)
            ax.xaxis.set_ticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticks(), rotation=rotation)

        mask_true = (
            dff[target] == self.target_true
            if target
            else np.full((len(dff)), True, dtype=bool)
        )

        ax_p = axs[1, 0]
        ax_n = axs[1, 1]

        if target == None or len(self.frames) == 1:
            ax_large = plt.subplot2grid((2, 3), (1, 0), colspan=2)
            ax_p = ax_n = ax_large

        sns.kdeplot(
            x=col,
            data=dff[mask_true],
            ax=ax_p,
            linewidth=2,
            label="Possitive Events" if target else None,  # hue will provide, if set
            hue="frame" if len(self.frames) > 1 else None,
            common_norm=False,
            fill=True,
            palette=pal,
        )
        ax_p.grid()

        sns.kdeplot(
            x=col,
            label="Negative Events" if target else None,  # hue will provide, if set
            data=dff[~mask_true],
            ax=ax_n,
            linewidth=2,
            hue="frame" if len(self.frames) > 1 else None,
            common_norm=False,
            fill=True,
            palette=pal,
        )
        ax_n.grid()

        sns.ecdfplot(
            x=col,
            hue="frame" if len(self.frames) > 1 else None,
            label="Possitive Events" if target else None,
            data=dff[mask_true],
            ax=axs[1, 2],
            linewidth=3,
            palette=pal,
        )  # , path_effects=path)
        axs[1, 2].grid()

        if (
            len(self.frames) == 1 and target
        ):  # will show the labels, otherwise not show hue? workaround
            ax_p.legend()
            ax_n.legend()
            axs[1, 2].legend()

        if len(self.frames) == 1 or target is None:
            tick(ax_large, "Distribution of Observations", 45)
            tick(axs[1, 2], "Cumulative Observations", 45)
        else:
            tick(ax_p, "Distribution of Positive Events", 45)
            tick(ax_n, "Distribution of Negative Events", 45)
            tick(axs[1, 2], "Cumulative Posstive Events", 45)

        fig.suptitle(
            "Distribution Analysis"
            + (" With target" if target else " Without Target")
            + " "
            + col,
            fontsize=25,
        )
        plt.show()

    def _plot_bins(self, col, pal):

        bins = self.get_bins_info_2(col)
        # len(bins['bins']) must be equal to len(frames)

        colors = sns.color_palette(pal, len(bins["bins"]))
        fig, ax1 = plt.subplots(figsize=(12, 8))

        dff = pd.DataFrame()
        for con, d in enumerate(bins["bins"]):
            r = d.copy()
            r["hue"] = self.frame_names[con] + " obs % per bin"
            r["indice"] = bins["splits_str"]
            r["bin_%"] = r["bin_%"] * 100
            dff = dff.append(pd.DataFrame(r), ignore_index=True)

        sns.barplot(x="indice", ax=ax1, hue="hue", y="bin_%", data=dff, palette=pal)

        ax1.set_xlabel("Bin ID", fontsize=12)
        ax1.set_ylabel("Observations Percentage Per Bin", fontsize=13)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_xticklabels(bins["splits_str"], rotation=60)

        ax2 = ax1.twinx()

        for e, bin in enumerate(bins["bins"]):
            value = bin["event_rate"]
            ax2.plot(
                bins["splits_str"],
                value,
                linestyle="solid",
                marker="o",
                color=colors[e],
                label=self.frame_names[e] + " event rate per bin",
                path_effects=[pe.Stroke(linewidth=2, foreground="black"), pe.Normal()],
            )

            mask_target = self.frames[e][self.target] == self.target_true
            mean_target = sum(mask_target) / len(mask_target)
            ax2.axhline(
                mean_target,
                ls=":",
                label=self.frame_names[e] + " mean event rate",
                color=colors[e],
            )
            ax2.set_ylabel("Event Rate", fontsize=13)

            mask_target_nan = self.frames[e][col].isna()
            if any(mask_target_nan):
                mask_target = (
                    self.frames[e][self.target] == self.target_true
                ) & ~mask_target_nan
                mean_target = sum(mask_target) / len(mask_target)
                ax2.axhline(
                    mean_target,
                    ls="--",
                    label=self.frame_names[e] + " mean event rate (w/o miss)",
                    color=colors[e],
                )

        h1, _ = ax1.get_legend_handles_labels()
        h2, _ = ax2.get_legend_handles_labels()

        ax1.set_title("Event rate by bin")
        ax1.legend(handles=h1 + h2, bbox_to_anchor=(1.1, 1), loc="upper left")

        fig.suptitle("Binning Analysis" + " " + col, fontsize=25)

        plt.show()

    def _get_cols(self, cols):

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]

            if not isinstance(cols, list):
                raise ValueError(
                    "Cols paramter must be either a string or a list of strings"
                )

        cols_to_iterate = (
            self.common_cols if not cols else self.common_cols.intersection(set(cols))
        )

        if len(cols_to_iterate) < 1:
            raise ValueError("There are no columns found in all frames")

        if cols is not None:
            not_found = set(cols) - set(cols_to_iterate)
            if len(not_found) > 0:
                raise ValueError("These columns were not found", list(not_found))

        return cols_to_iterate

    def show_distribution_plots(
        self, cols=None, palette="husl", exclude_target=False, sd_cutoff=2.5
    ):
        cols_to_iterate = self._get_cols(cols)

        if exclude_target is True and self.target is None:
            raise ValueError("No target provided in init to exclude")

        target_to_use = self.target if not exclude_target else None

        for c in cols_to_iterate:
            self._plot_individual_distributions(
                col=c, pal=palette, target=target_to_use, sd_cutoff=sd_cutoff
            )

    def show_binning_plots(self, cols=None, palette="husl"):
        cols_to_iterate = self._get_cols(cols)

        if self.target is None:
            raise ValueError("No target provided in init. Needed to plot the bins")

        for c in cols_to_iterate:
            self._plot_bins(col=c, pal=palette)

    def get_analysis_columns(self, estimator=None, cols=None):

        cols_to_iterate = self._get_cols(cols)

        for y, c in enumerate(cols_to_iterate):

            print("-----------\n")
            print("Column: ", c)

            analysis = {}
            info = {}
            for x, f in enumerate(self.frames):

                info["Total rows"] = len(f[c])
                info["# Nans"] = f[c].isna().sum()
                info["% Nans"] = info["# Nans"] / info["Total rows"]
                info["Unique Values"] = f[c].nunique()
                info["Data type"] = f[c].dtype
                info["Max / Min"] = str(f[c].max()) + " / " + str(f[c].min())

                analysis[self.frame_names[x]] = info.copy()

            print(pd.DataFrame(analysis))
