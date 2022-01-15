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

    def plot_density(self, col):

        nrows = len(self.frames)
        fig, axs = plt.subplots(ncols=2, nrows=nrows, figsize=(15, 5))
        colors = sns.color_palette(None, len(self.frame_names))

        axs[0].set_title("Possitive cases")
        axs[1].set_title("Negative cases")

        # ###
        # dff = pd.DataFrame()
        # for e, f in enumerate(self.frames):
        #     a = f[[col, self.target]].copy()
        #     a["hue"] = self.frame_names[e]
        #     dff = dff.append(
        #         a.loc[a[self.target] == self.target_true],
        #         ignore_index=True,
        #     )

        # sns.histplot(
        #     data=dff,
        #     x=col,
        #     hue="hue",
        #     # kde=True,
        #     stat="density",
        #     ax=axs[2],
        #     alpha=0.5,
        #     # shade=True,
        # )

        ###
        for x, f in enumerate(self.frames):
            sns.kdeplot(
                f.loc[f[self.target] == self.target_true][col],
                label=self.frame_names[x] + "__" + str(self.target_true),
                color=colors[x],
                ax=axs[0],
            )
            axs[0].axvline(
                f.loc[f[self.target] == self.target_true][col].mean(),
                ls="--",
                color=colors[x],
            )
            axs[0].axvline(
                f.loc[f[self.target] == self.target_true][col].mode()[0],
                ls=":",
                color=colors[x],
            )

            sns.kdeplot(
                f.loc[f[self.target] != self.target_true][col],
                label=self.frame_names[x],
                color=colors[x],
                ax=axs[1],
            )
            axs[1].axvline(
                f.loc[f[self.target] != self.target_true][col].mean(),
                ls="--",
                color=colors[x],
            )
            axs[1].axvline(
                f.loc[f[self.target] != self.target_true][col].mode()[0],
                ls=":",
                color=colors[x],
            )

        line_solid = mlines.Line2D([], [], color="black", linestyle="--", label="Mean")
        line_dashed = mlines.Line2D([], [], color="black", linestyle=":", label="Mode")

        handles, labels = plt.gca().get_legend_handles_labels()

        plt.legend(handles=handles + [line_dashed] + [line_solid])
        plt.show()

    def plot_binning(self, col):

        bins = self.get_bins_info_2(col)
        # len(bins['bins']) must be equal to len(frames)

        colors = sns.color_palette(None, len(bins["bins"]))
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # there should be a more elegant way of doing this
        # Tried with matplotlib directly but it was too complicated to center things
        # So I'm creating a dataframe just to plot the barplot
        dff = pd.DataFrame()
        for con, d in enumerate(bins["bins"]):
            r = d.copy()
            r["hue"] = self.frame_names[con] + " obs % per bin"
            r["indice"] = bins["splits_str"]
            r["bin_%"] = r["bin_%"] * 100
            dff = dff.append(pd.DataFrame(r), ignore_index=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="indice", ax=ax1, hue="hue", y="bin_%", data=dff)

        # #barWidth = 0.25

        # for e, bin in enumerate(bins['bins']):
        #     print(bin)
        #     value = bin['bin_%'] * 100

        #     if e==0:
        #          r = np.arange(len(value))
        #     else:
        #         r = [x + width for x in r]

        #     ax1.bar(r, value, width, color=colors[e], label=frame_names[e] + " obs % per bin")

        ax1.set_xlabel("Bin ID", fontsize=12)
        ax1.set_ylabel("Observations Percentage Per Bin", fontsize=13)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_xticklabels(bins["splits_str"], rotation=60)
        # plt.xticks(rotation=60)
        # plt.xticks([r + width for r in range(len(value))], bins['splits_str'])

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

        # line_solid = mlines.Line2D([], [], color='black', linestyle='--', label='Mean')
        # line_dashed = mlines.Line2D([], [], color='black', linestyle=':',  label='Mode')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()

        ax1.set_title(col + " binning analysis (event rate)", fontsize=20)
        ax1.legend(handles=h1 + h2, bbox_to_anchor=(1.1, 1), loc="upper left")
        plt.show()

    def get_analysis_columns(self, estimator=None, cols=None):

        cols_to_iterate = (
            self.common_cols if not cols else self.common_cols.intersection(set(cols))
        )

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

            self.plot_density(c)
            self.plot_binning(c)
