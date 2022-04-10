import pandas as pd

from instrumentum.analysis.dashboards import (
    dashboard_categorical_with_binary_target,
    dashboard_continuos_with_binary_target,
)


class ColumnType:
    CONTINUOS = "continuos"
    # Nominal and Ordinal
    CATEGORY_GENERAL = "category_general"
    # Binary
    CATEGORY_BINARY = "category_binary"


class DistAnalyzer:
    def __init__(self, df, y=None, y_true=1, y_type=None, cluster=None):

        self.df = df

        # Y validation -------
        if y is not None and y not in df.columns.to_list():
            raise ValueError("y must be included in df")

        self.y = y

        # Y_type validation ------
        if y is None and y_type is not None:
            raise ValueError("if y_type is specified, y must be specified as well")

        if y is not None and y_type is None:
            y_type = self._get_col_type(self.y)

        self.y_type = y_type

        # Y_true validation
        if y is None and y_true is not None:
            raise ValueError("if y_true is specified, y must be specified as well")

        if y is not None and y_true is not None and y_true not in df[y].unique():
            raise ValueError("Y true value not found as a value of df[y")

        self.y_true = y_true

        # cluster validation
        if cluster is not None and cluster not in df.columns.to_list():
            raise ValueError("cluster must be included in df")

        self.cluster = cluster

        # cols = [f.columns.tolist() for f in frames]
        # self.common_cols = set(cols[0]).intersection(*cols)

        # if frames_names:
        #     self.frame_names = frames_names
        # else:
        #     self.frame_names = ["Dataset " + str(x) for x in range(len(frames))]

        # if target and target not in self.common_cols:
        #     raise ValueError("Target not included in all dataframes")

    def _get_cols(self, cols):

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]

            if not isinstance(cols, list):
                raise ValueError(
                    "Cols paramter must be either a string or a list of strings"
                )

        return cols

    def _get_col_type(self, col):

        # TODO: maybe do this for each cluster?
        nunique = self.df[col].nunique()

        if nunique <= 2:
            print("Column looks binary")
            return ColumnType.CATEGORY_BINARY

        if nunique < 20 or nunique / self.df[col].count() < 0.01:
            print("Column looks categorical")
            return ColumnType.CATEGORY_GENERAL

        print("Looks continuos")
        return ColumnType.CONTINUOS

    def show_dashboard(self, xs=None, palette="husl"):

        for x in self._get_cols(xs):

            # get_type_of_column
            x_type = self._get_col_type(x)

            # X is continuos, and Y is binary
            if (
                x_type == ColumnType.CONTINUOS
                and self.y_type == ColumnType.CATEGORY_BINARY
            ):
                dashboard_continuos_with_binary_target(
                    self.df, x=x, y=self.y, cluster=self.cluster
                )

            # X is categorical, and Y is binary
            elif (
                x_type in [ColumnType.CATEGORY_BINARY, ColumnType.CATEGORY_GENERAL]
                and self.y_type == ColumnType.CATEGORY_BINARY
            ):
                dashboard_categorical_with_binary_target(
                    self.df, x=x, y=self.y, cluster=self.cluster
                )

    # def get_analysis_columns(self, estimator=None, cols=None):

    #     cols_to_iterate = self._get_cols(cols)

    #     for y, c in enumerate(cols_to_iterate):

    #         print("-----------\n")
    #         print("Column: ", c)

    #         analysis = {}
    #         info = {}
    #         for x, f in enumerate(self.frames):

    #             info["Total rows"] = len(f[c])
    #             info["# Nans"] = f[c].isna().sum()
    #             info["% Nans"] = info["# Nans"] / info["Total rows"]
    #             info["Unique Values"] = f[c].nunique()
    #             info["Data type"] = f[c].dtype
    #             info["Max / Min"] = str(f[c].max()) + " / " + str(f[c].min())

    #             analysis[self.frame_names[x]] = info.copy()

    #         print(pd.DataFrame(analysis))
