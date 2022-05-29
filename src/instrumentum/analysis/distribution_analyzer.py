from scipy.stats import normaltest

from instrumentum.analysis.dashboards import (
    dashboard_categorical_with_binary_target,
    dashboard_continuos_with_binary_target,
)


# 2 Standard Deviations from the Mean: 95%
def remove_outliers(df, col, sd_cutoff=2, qr_cutoff=[1.5, 1.5]):

    _, p = normaltest(df[~df[col].isna()][col])

    if p > 0.2:  # quite permissive
        print(col + " looks normal. Using Standard Deviation")
        outliers = abs(df[col]) >= df[col].mean() + df[col].std() * sd_cutoff

    else:
        print(col + " looks skewed. Using quartiles")
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        outliers = (df[col] < (Q1 - qr_cutoff[0] * IQR)) | (
            df[col] > (Q3 + qr_cutoff[1] * IQR)
        )

    n_removed = len(df) - len(df[~outliers])
    print("Outliers removed: ", n_removed, " %: ", n_removed / len(df))

    return df[~outliers]


class ColumnType:
    CONTINUOS = "continuos"
    # Nominal and Ordinal
    CATEGORY_GENERAL = "category_general"
    # Binary
    CATEGORY_BINARY = "category_binary"


class DistAnalyzer:
    def __init__(self, df, y=None, y_true=None, y_type=None, cluster=None):

        self.df = df

        # Y validation -------
        if y is not None and y not in df.columns.to_list():
            raise ValueError("y must be included in df")

        self.y = y

        # Y_type validation ------
        if y is None and y_type is not None:
            raise ValueError(
                "if y_type is specified, y must be specified as well"
            )

        if y is not None and y_type is None:
            y_type = self._get_col_type(self.y)

        self.y_type = y_type

        # Y_true validation
        if y is None and y_true is not None:
            raise ValueError(
                "if y_true is specified, y must be specified as well"
            )

        if (
            y is not None
            and y_true is not None
            and y_true not in df[y].unique()
        ):
            raise ValueError("Y true value not found as a value of df[y")

        self.y_true = y_true

        # cluster validation
        if cluster is not None and cluster not in df.columns.to_list():
            raise ValueError("cluster must be included in df")

        self.cluster = cluster

    def _get_cols(self, cols):

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]

            if not isinstance(cols, list):
                raise ValueError(
                    "Cols paramter must be either a string or a list of "
                    + "strings"
                )

        return cols

    def _get_col_type(self, col):

        # TODO: maybe do this for each cluster?
        nunique = self.df[col].nunique()

        if nunique <= 2:
            print(col + " looks binary")
            return ColumnType.CATEGORY_BINARY

        if nunique < 20 or nunique / self.df[col].count() < 0.01:
            print(col + " looks categorical")
            return ColumnType.CATEGORY_GENERAL

        print(col + " looks continuos")
        return ColumnType.CONTINUOS

    def show_dashboard(self, xs=None, keep_outliers=True, palette="husl"):

        for x in self._get_cols(xs):

            # Let's create a copy to make sure the original is not affected.
            # Only use important columns
            df = self.df[
                self.df.columns.intersection([x, self.y, self.cluster])
            ].copy()

            # get_type_of_column
            x_type = self._get_col_type(x)

            if not keep_outliers and x_type == ColumnType.CONTINUOS:
                df = remove_outliers(df, x)

            # X is continuos, and Y is binary
            if (
                x_type == ColumnType.CONTINUOS
                and self.y_type == ColumnType.CATEGORY_BINARY
            ):
                dashboard_continuos_with_binary_target(
                    df, x=x, y=self.y, cluster=self.cluster
                )

            # X is categorical, and Y is binary
            elif (
                x_type
                in [ColumnType.CATEGORY_BINARY, ColumnType.CATEGORY_GENERAL]
                and self.y_type == ColumnType.CATEGORY_BINARY
            ):
                dashboard_categorical_with_binary_target(
                    df, x=x, y=self.y, cluster=self.cluster
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
