########################################################################################################################
# Packages
########################################################################################################################

# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from joblib import Parallel, delayed
import copy
import warnings
import time
from typing import Union, Literal
from inspect import getargspec

# Scikit
from sklearn.metrics import (make_scorer, roc_auc_score, accuracy_score, roc_curve, confusion_matrix,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import cross_val_score, GridSearchCV, check_cv, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils import _safe_indexing
from sklearn.base import BaseEstimator, TransformerMixin, clone  # ClassifierMixin

# ML
import xgboost as xgb
import lightgbm as lgbm
from itertools import product  # for GridSearchCV_xlgb

# Other
from scipy.interpolate import splev, splrep
from scipy.cluster.hierarchy import linkage
from pycorrcat.pycorrcat import corr as corrcat
from statsmodels.nonparametric.smoothers_lowess import lowess


########################################################################################################################
# General Functions
########################################################################################################################

# --- General ----------------------------------------------------------------------------------------------------------


def debugtest(a=1, b=2):
    print(a)
    print(b)
    print("blub")
    # print("blub2")
    # print("blub3")
    print(a)
    print(b)
    print("kasgholiahg")
    return "done"


def tmp(a: pd.DataFrame) -> list:
    """
    Print summary of (categorical) varibles (similar to R's summary function)

    Parameters
    ----------
    par1: Dataframe 
        Dataframe comprising columns to be summarized
    par2: int
        Restrict number of member listings 

    Returns
    ------- 
    dataframe which comprises summary of variables
    """
    pass


def kwargs_reduce(kwargs: dict, function) -> dict:
    """ Reduce kwargs dict to the arguments of function """
    return {key: value for key, value in kwargs.items() if key in getargspec(function).args}


def diff(a, b) -> Union[list, np.ndarray]:
    """ Creates setdiff i.e. a_not_in_b, for arrays or lists """
    a_not_b = np.setdiff1d(a, b, True)
    if (type(a) is list and type(b) is list):
        a_not_b = list(a_not_b)
    return a_not_b


def add(a, b) -> list:
    """
    Provide concatenation of suffix or prefix to every list element, similar to "a"+b or a+"b" for object arrays
    """
    if type(a) is str:
        return [a + x for x in b]
    if type(b) is str:
        return [x + b for x in a]


def interleave(a, b):
    """ Interleaves two lists, i.e. returns [a[0], b[0], a[1], b[1], ...] """
    return [x for tup in zip(a, b) for x in tup]


'''
def logit(p: float) -> float:
    return(np.log(p / (1 - p)))


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))
'''


def show_figure(fig):
    """ Creates a dummy figure and uses its manager to display closed 'fig' """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


# Plot list of tuples (plot_call, kwargs)
def plot_l_calls(l_calls, n_cols=2, n_rows=2, figsize=(16, 10), pdf_path=None, constrained_layout=False):
    """
    Plot list of tuples (plot_function, {kwargs}) with plot_functions that must have 'ax' parameter like
    seaborn or pandas plot and kwargs representing function parameters

    Parameters
    ----------
    l_calls: list 
        List of tuples, e.g.
        [(sns.histplot, {"x": np.random.normal(size=100), "kde": True}),
         (sns.lineplot, {"x": np.arange(10), "y": np.arange(10)})]
    n_cols, n_rows: int
        Number of cols and rows in grid layout
    figsize: tuple(int, int)
        Width and height of plot in inches
    pdf_path: str
        Location of pdf to print to (optional)
    constrained_layout: bool
        Use as alternative to tight_layout?

    Returns
    ------- 
    List of pages as tuples (figure of page, axes of figure), i.e. each tuple is similar to what plt.subplots() returns
    """

    # Build pages
    l_pages = list()
    for i, (plot_function, kwargs) in enumerate(l_calls):
        # Init new page
        if i % (n_rows * n_cols) == 0:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=constrained_layout)
            l_pages.append((fig, axes))
            i_ax = 0

        # Plot call
        plot_function(ax=axes.flat[i_ax] if (n_rows * n_cols > 1) else axes, **kwargs)
        i_ax += 1

        # "Close" page
        if (i_ax == n_rows * n_cols) or (i == len(l_calls) - 1):
            # Remove unused axes
            if (i == len(l_calls) - 1):
                for k in range(i_ax, n_rows * n_cols):
                    axes.flat[k].axis("off")

            # Constrained or tight layout?
            if constrained_layout:
                fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.1, wspace=0.1)
            else:
                fig.tight_layout()

    # Write pdf
    if pdf_path is not None:
        with PdfPages(pdf_path) as pdf:
            for page in l_pages:
                pdf.savefig(figure=page[0], bbox_inches="tight", pad_inches=0.2)

    return l_pages


# --- Metrics for sklearn scorer ---------------------------------------------------------------------------------------

# Regr

def reduce2prob(yhat):
    """ Reduce prediction matrix to 1-dim-array comprising probability of "1"-class """
    if (yhat.ndim == 2) and (yhat.shape[1] == 2):
        return yhat[:, 1]
    else:
        return yhat


def spear(y, yhat) -> float:
    """ Spearman correlation (working also for classification tasks) """
    yhat = reduce2prob(yhat)  # Catch classification case
    spear = pd.DataFrame({"y": y, "yhat": yhat}).corr(method="spearman").values[0, 1]
    return spear


def pear(y, yhat) -> float:
    """ Pearson correlation (working also for classification tasks) """
    yhat = reduce2prob(yhat)  # Catch classification case
    pear = pd.DataFrame({"y": y, "yhat": yhat}).corr(method="pearson").values[0, 1]
    return pear


def rmse(y, yhat) -> float:
    """ Root mean squared error """
    return np.sqrt(np.mean(np.power(y - yhat, 2)))


def ame(y, yhat) -> float:
    """ Absolute mean error"""
    return np.abs(np.mean(y - yhat))


def amde(y, yhat) -> float:
    """ Absolute median error"""
    return np.abs(np.median(y - yhat))


def mae(y, yhat) -> float:
    """ Mean absolute error """
    return np.mean(np.abs(y - yhat))


def mdae(y, yhat) -> float:
    """ Median absolute error """
    return np.median(np.abs(y - yhat))


def smape(y, yhat) -> float:
    """ Symmetric mean absolute percentage error """
    return np.mean(200 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat)))


def smdape(y, yhat) -> float:
    """ Symmetric median absolute percentage error """
    return np.median(200 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat)))


# Class + Multiclass

def auc(y, yhat) -> float:
    """ AUC (working also for regression task which is then basically the concordance) """

    # Usually yhat is a matrix, so reduce it to 1-dim array
    yhat = reduce2prob(yhat)

    # Regression case
    if (y.ndim == 1) & (type_of_target(y) == "continuous"):
        yhat = MinMaxScaler().fit_transform(yhat.reshape(-1, 1))[:, 0]
        y = MinMaxScaler().fit_transform(y)

    return roc_auc_score(y, yhat, multi_class="ovr")


def acc(y, yhat):
    """ Accuracy """
    if yhat.ndim > 1:
        yhat = yhat.argmax(axis=1)
    if y.ndim > 1:
        y = pd.DataFrame(y).values.argmax(axis=1)
    return accuracy_score(y, yhat)


# Standard scoring metrics
D_SCORER = {"REGR": {"spear": make_scorer(spear, greater_is_better=True),
                     "rmse": make_scorer(rmse, greater_is_better=False),
                     "ame": make_scorer(ame, greater_is_better=False),
                     "mae": make_scorer(mae, greater_is_better=False)},
            "CLASS": {"auc": make_scorer(auc, greater_is_better=True, needs_proba=True),
                      "acc": make_scorer(acc, greater_is_better=True)},
            "MULTICLASS": {"auc": make_scorer(auc, greater_is_better=True, needs_proba=True),
                           "acc": make_scorer(acc, greater_is_better=True)}}


########################################################################################################################
# Explore
########################################################################################################################

# --- Non-plots --------------------------------------------------------------------------------------------------------

# Overview of values
def value_counts(df, topn=5, dtypes=None):
    """
    Print summary of (categorical) varibles (similar to R's summary function)

    Parameters
    ----------
    df: Dataframe 
        Dataframe comprising columns to be summarized
    topn: integer
        Restrict number of member listings
    dtype: list or None
        None or list of dtypes (e.g. ["object"]) to filter. 

    Returns
    ------- 
    Dataframe which comprises summary of variables
    """
    if dtypes is not None:
        df_select = df.select_dtypes(dtypes)
    else:
        df_select = df
    df_return = pd.concat([(df_select[catname].value_counts().reset_index()
                            .pipe(lambda x: x.rename(columns={"index": f"{catname} ({str(len(x))})", catname: ""}))
                            .iloc[:topn, :])
                           for catname in df_select.columns.values],
                          axis=1).fillna("")
    return df_return


# Binning with correct label
def bin(feature, n_bins=5, precision=3) -> pd.Series:
    """
    Bins a numeric feature and assigns a verbose label, describing the interval range of the bin

    Parameters
    ----------
    feature: Numpy array or Pandas series
        Feature to bin
    n_bins: int
        Number of bins
    precision: int
        Precision of interval labels

    Returns
    -------
    String series
    """
    # Transform, e.g. to "q0 (left-border, right-border)"
    feature_binned = pd.qcut(feature, n_bins, duplicates="drop", precision=precision)
    categories = feature_binned.cat.categories  # save
    feature_binned = "q" + feature_binned.cat.codes.astype("str") + " " + feature_binned.astype("str")

    # Replace first category starting interval string with correct label
    feature_binned = feature_binned.str.replace("\(" + str(categories[0].left),
                                                "[" + str(pd.Series(feature).min().round(precision)))

    # Reassign missings
    feature_binned = feature_binned.replace({"q-1 nan": np.nan})

    return feature_binned


# Univariate model performance
def variable_performance(feature, target, scorer, target_type=None, splitter=KFold(5), groups=None, verbose=True):
    """
    Calculates univariate variable performance by applying LinearRegression or LogisticRegression (depending on target)
    on single feature model and calcuating scoring metric.\n 
    "Categorical" features are 1-hot encoded, numeric features are binned into 5 bins (in order to approximate 
    also nonlinear effect)

    Parameters
    ----------
    feature: Numpy array or Pandas series
        Feature for which to calculate importance
    target: Numpy array or Pandas series
        Target variable
    scorer: sklearn.metrics scorer        
    target_type: "CLASS", "REGR", "MULTICLASS", None, default=None
        Overwrites function's determination of target type (if not None)
    splitter: sklearn.model_selection splitter
    groups: Numpy array or Pandas series
        Grouping variable in case of using Grouped splitter
    verbose: Boolean
        Print processing information

    Returns
    -------
    Numeric value representing scoring result
    """

    # Drop all missings
    df_hlp = pd.DataFrame().assign(feature=feature, target=target)
    if groups is not None:
        df_hlp["groups_for_split"] = groups
    df_hlp = df_hlp.dropna().reset_index(drop=True)

    # Detect types
    if target_type is None:
        target_type = dict(continuous="REGR", binary="CLASS",
                           multiclass="MULTICLASS")[type_of_target(df_hlp["target"])]
    numeric_feature = pd.api.types.is_numeric_dtype(df_hlp["feature"])
    if verbose:
        print("Calculate univariate performance for",
              "numeric" if numeric_feature else "categorical",
              "feature '" + feature.name + "' for " + target_type + " target '" + target.name + "'")

    # Calc performance
    perf = np.mean(cross_val_score(
        estimator=(LinearRegression() if target_type == "REGR" else LogisticRegression()),
        X=(KBinsDiscretizer().fit_transform(df_hlp[["feature"]]) if numeric_feature else
           OneHotEncoder().fit_transform(df_hlp[["feature"]])),
        y=df_hlp["target"],
        cv=splitter.split(df_hlp, groups=df_hlp["groups_for_split"] if groups is not None else None),
        scoring=scorer))

    return perf


# Winsorize
class Winsorize(BaseEstimator, TransformerMixin):
    """
    Winsorizing transformer for clipping outlier

    Parameters
    ----------
    lower_quantile: float, default=None
        Lower quantile (which must be between 0 and 1) at which to clip all columns
    upper_quantile: float, default=None
        Upper quantile (which must be between 0 and 1) at which to clip all columns

    Attributes
    ----------
    a_lower_: array of lower quantile values   
    a_upper_: array of upper quantile values  
    """

    def __init__(self, lower_quantile=None, upper_quantile=None):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        #self._private_attr = whatever

    def fit(self, X, *_):
        X = pd.DataFrame(X)
        if self.lower_quantile is not None:
            self.a_lower_ = np.nanquantile(X, q=self.lower_quantile, axis=0)
        else:
            self.a_lower_ = None
        if self.upper_quantile is not None:
            self.a_upper_ = np.nanquantile(X, q=self.upper_quantile, axis=0)
        else:
            self.a_upper_ = None
        return self

    def transform(self, X, *_):
        if (self.lower_quantile is not None) or (self.upper_quantile is not None):
            X = np.clip(X, a_min=self.a_lower_, a_max=self.a_upper_)
        return X


# Map Non-topn frequent members of a string column to "other" label
class Collapse(BaseEstimator, TransformerMixin):
    """
    Collapsing transformer mapping non-topN frequent member of a categorical feature to "other" label 

    Parameters
    ----------
    n_top: int, default=10
        Number of topN members that are not collapsed
    other_label: str, default="_OTHER_"
        Collective member name to which the non-topN frequent members are mapped ("collapsed")

    Attributes
    ----------
    d_top_: dictionary comprising per feature the names of the topN members
    """

    def __init__(self, n_top=10, other_label="_OTHER_"):
        self.n_top = n_top
        self.other_label = other_label

    def fit(self, X, *_):
        self.d_top_ = pd.DataFrame(X).apply(lambda x: x.value_counts().index.values[:self.n_top])
        return self

    def transform(self, X):
        X = pd.DataFrame(X).apply(lambda x: x.where(np.in1d(x, self.d_top_[x.name]),
                                                    other=self.other_label)).values
        return X


# Impute Mode (simpleimputer is too slow)
class ImputeMode(BaseEstimator, TransformerMixin):
    """
    Imputing transformer for categorical feature which imputes the mode member (as scikit's simpleimputer is too slow)

    Parameters
    ----------
    None

    Attributes
    ----------
    impute_values_: dictionary comprising per feature the label of the most frequent member
    """

    def __init__(self):
        pass

    def fit(self, X):
        self.impute_values_ = pd.DataFrame(X).mode().iloc[0].to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).fillna(self.impute_values_).values
        return X


# --- Plots ------------------------------------------------------------------------------------------------------------

def _helper_calc_barboxwidth(feature, target, min_width=0.2):
    """ 
    Helper function calculating widht of boxes or length of bars reflecting feature distribution used by
    plot_cate_CLASS, plot_cate_MULTICLASS, plot_cate_REGR and plot_pd. 
    Returns dataframe with corresponding information.
    """
    df_hlp = pd.crosstab(feature, target)
    df_barboxwidth = (df_hlp.div(df_hlp.sum(axis=1), axis=0)
                      .assign(w=df_hlp.sum(axis=1))
                      .reset_index()
                      .assign(pct=lambda z: 100 * z["w"] / df_hlp.values.sum())
                      .assign(w=lambda z: 0.9 * z["w"] / z["w"].max())
                      .assign(**{feature.name + "_fmt":
                                 lambda z: z[feature.name] + z["pct"].map(lambda x: f" ({x:0.1f}%)")})
                      .assign(w=lambda z: np.where(z["w"] < min_width, min_width, z["w"])))
    return df_barboxwidth


def _helper_adapt_feature_target(feature, target, feature_name, target_name, verbose=True) -> tuple:
    """ 
    Helper function transforming feature and target array into series with appropriate name with corresponding 
    missings removed in both sereis, used by all plot_nume/cate_CLASS/MULTICLASS/REGR plot functions. 
    Returns tuple of (feature, target, pct_miss_feature)
    """
    # Convert to Series and adapt labels
    if not isinstance(feature, pd.Series):
        feature = pd.Series(feature)
        feature.name = feature_name if feature_name is not None else "x"
    if feature_name is not None:
        feature = feature.copy()
        feature.name = feature_name
    if not isinstance(target, pd.Series):
        target = pd.Series(target)
        target.name = target_name if target_name is not None else "y"
    if target_name is not None:
        target = target.copy()
        target.name = target_name

    if verbose:
        print("Plotting " + feature.name + " vs. " + target.name)

    # Remove missings
    mask_target_miss = target.isna()
    n_target_miss = mask_target_miss.sum()
    if n_target_miss:
        target = target[~mask_target_miss]
        feature = feature[~mask_target_miss]
        print("ATTENTION: " + str(n_target_miss) + " records removed due to missing target!")
        mask_target_miss = target.notna()
    mask_feature_miss = feature.isna()
    n_feature_miss = mask_feature_miss.sum()
    pct_miss_feature = 100 * n_feature_miss / len(feature)
    if n_feature_miss:
        target = target[~mask_feature_miss]
        feature = feature[~mask_feature_miss]
        #warnings.warn(str(n_feature_miss) + " records removed due to missing feature!")

    return (feature, target, pct_miss_feature)


def _helper_inner_barplot(ax, x, y, inset_size=0.2):
    """ Helper function creating barplot, used as inner distribution plot by all plot_cate_xxx plots """
    # Memorize ticks and limits
    xticks = ax.get_xticks()
    xlim = ax.get_xlim()

    # Create space
    ax.set_xlim(xlim[0] - 1.2 * inset_size * (xlim[1] - xlim[0]), xlim[1])

    # Create shared inset axis
    inset_ax = ax.inset_axes([0, 0, inset_size, 1])
    # inset_ax.get_xaxis().set_visible(False)
    # inset_ax.get_yaxis().set_visible(False)
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])
    ax.get_shared_y_axes().join(ax, inset_ax)

    # Plot
    inset_ax.barh(x, y,
                  color="lightgrey", edgecolor="black", linewidth=1)

    # More space for plot
    left, right = inset_ax.get_xlim()
    inset_ax.set_xlim(left, right * 1.2)

    # Border
    inset_ax.axvline(inset_ax.get_xlim()[1], color="black")

    # Remove senseless ticks
    yticks = inset_ax.get_yticks()
    if len(yticks) > len(x):
        _ = inset_ax.set_yticks(yticks[1::2])
    _ = ax.set_xticks(xticks[(xticks >= xlim[0]) & (xticks <= xlim[1])])


def _helper_inner_barplot_rotated(ax, x, y, inset_size=0.2):
    """ 
    Helper function completely similar to _helper_inner_barplot but which rotated barplot, that can be
    plotted on x-axis (in fact every "x"-function call is replaced by "y"-function call) 
    """

    # Memorize ticks and limits
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()

    # Create space
    ax.set_ylim(ylim[0] - 1.2 * inset_size * (ylim[1] - ylim[0]), ylim[1])

    # Create shared inset axis
    inset_ax = ax.inset_axes([0, 0, 1, inset_size])
    inset_ax.set_yticklabels([])
    inset_ax.set_xticklabels([])
    ax.get_shared_x_axes().join(ax, inset_ax)

    # Plot
    inset_ax.bar(x, y,
                 color="lightgrey", edgecolor="black", linewidth=1)

    # More space for plot
    left, right = inset_ax.get_ylim()
    inset_ax.set_ylim(left, right * 1.2)

    # Border
    inset_ax.axhline(inset_ax.get_ylim()[1], color="black")

    # Remove senseless ticks
    xticks = inset_ax.get_xticks()
    if len(xticks) > len(y):
        _ = inset_ax.set_xticks(xticks[1::2])
    _ = ax.set_yticks(yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])])


def plot_cate_CLASS(ax,
                    feature, target,
                    feature_name=None, target_name=None,
                    target_category=None,
                    target_lim=None,
                    min_width=0.2, inset_size=0.2, refline=True,
                    title=None,
                    add_miss_info=False,
                    color=list(sns.color_palette("colorblind").as_hex()),
                    verbose=True):
    """
    Plots categorical feature vs classfication target (as bars).

    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    feature: Numpy array or Pandas series
        Categorical feature (must be of dtype object or str) to plot on y-axis.
    target: Numpy array or Pandas series
        Classifiction target to plot on x-axis.
    feature_name: str (optional)
        If specified used as feature's name/label in plot.
    target_name: str (optional)
        If specified used as target's name/label in plot.
    target_category: str or int (optional)
        If not specified the minority class is used and plotted.
    target_lim: 2-tuple of float or int
        Limits the y-axis on which the target is plotted.
    min_width: float
        Minimum width of bars. Per default the feature's member frequency determines the width per member
        which might end with a "line". 
        Can be orriden by this paramter and set to a minimum.
    inset_size: float
        Relative size of distribution bar plot on y-axis.
    refline: boolean
        Print a reference line representing the base rate.
    title: str
        Title of plot
    add_miss_info: boolean
        Show percentage of missings in feature's axis label.
    color: list or str
        Color used for bars. If list provided the second item is used.
    verbose: Boolean
        Print processing information.

    Returns
    -------
    Nothing
    """
    # Adapt feature and target
    feature, target, pct_miss_feature = _helper_adapt_feature_target(feature, target, feature_name, target_name,
                                                                     verbose=verbose)

    # Adapt color
    if isinstance(color, list):
        color = color[1]

    # Add title
    if title is None:
        title = feature.name

    # Get "1" class
    if target_category is None:
        target_category = target.value_counts().sort_values().index.values[0]

    # Prepare data
    df_plot = _helper_calc_barboxwidth(feature, target, min_width=min_width)

    # Barplot
    ax.barh(df_plot[feature.name + "_fmt"], df_plot[target_category], height=df_plot["w"],
            color=color, edgecolor="black", alpha=0.5, linewidth=1)
    ax.set_xlabel("avg(" + target.name + ")")
    ax.set_title(title)
    if target_lim is not None:
        ax.set_xlim(target_lim)

    # Refline
    if refline:
        ax.axvline((target == target_category).sum() / len(target), linestyle="dotted", color="black")

    # Inner barplot
    _helper_inner_barplot(ax, x=df_plot[feature.name + "_fmt"], y=df_plot["pct"], inset_size=inset_size)

    # Missing information
    if add_miss_info:
        ax.set_ylabel(feature.name)
        ax.set_ylabel(ax.get_ylabel() + f" ({pct_miss_feature:0.1f} % NA)")


def plot_cate_MULTICLASS(ax,
                         feature, target,
                         feature_name=None, target_name=None,
                         min_width=0.2, inset_size=0.2, refline=True,
                         title=None,
                         add_miss_info=True,
                         color=list(sns.color_palette("colorblind").as_hex()),
                         reverse=False,
                         exchange_x_y_axis=False,
                         verbose=True):
    """
    Plots categorical feature vs multiclass target (as segemented bars).

    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    feature: Numpy array or Pandas series
        Categorical feature (must be of dtype object or str) to plot on y-axis.
    target: Numpy array or Pandas series
        Multiclass target to plot on x-axis.
    feature_name: str (optional)
        If specified used as feature's name/label in plot.
    target_name: str (optional)
        If specified used as target's name/label in plot.
    min_width: float
        Minimum width of bars. Per default the feature's member frequency determines the width per member
        which might end with a "line", which would prevent from distinguishing target members. 
        Can be orriden by this paramter and set to a minimum.
    inset_size: float
        Relative size of distribution bar plot on y-axis.
    refline: boolean
        Print a reference line representing the base rate.
    title: str
        Title of plot.
    add_miss_info: boolean
        Show percentage of missings in feature's axis label.
    color: list
        List of colors assigned to target's categories.
    reverse: boolean
        Should bar segments be reversed?
    exchange_x_y_axis: boolean
        Should default horizontal bars on y-Axis be plotted as vertical bars on x-Axis?
    verbose: Boolean
        Print processing information.

    Returns
    -------
    Nothing
    """
    # Adapt feature and target
    feature, target, pct_miss_feature = _helper_adapt_feature_target(feature, target, feature_name, target_name,
                                                                     verbose=verbose)

    # Add title
    if title is None:
        title = feature.name

    # Prepare data
    df_plot = _helper_calc_barboxwidth(feature, target, min_width=min_width)

    # Reverse
    if reverse:
        df_plot = df_plot.iloc[::-1]

    # Segmented barplot
    offset = np.zeros(len(df_plot))
    for m, member in enumerate(np.sort(target.unique())):
        if not exchange_x_y_axis:
            ax.barh(y=df_plot[feature.name + "_fmt"], width=df_plot[member], height=df_plot["w"],
                    left=offset,
                    color=color[m], label=member, edgecolor="black", alpha=0.5, linewidth=1)
        else:
            ax.bar(x=df_plot[feature.name + "_fmt"], height=df_plot[member], width=df_plot["w"],
                   bottom=offset,
                   color=color[m], label=member, edgecolor="black", alpha=0.5, linewidth=1)
        offset = offset + df_plot[member].values
        ax.legend(title=target.name, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title)

    # Inner barplot
    if not exchange_x_y_axis:
        _helper_inner_barplot(ax, x=df_plot[feature.name + "_fmt"], y=df_plot["pct"], inset_size=inset_size)
    else:
        _helper_inner_barplot_rotated(ax, x=df_plot[feature.name + "_fmt"], y=df_plot["pct"], inset_size=inset_size)

    # Missing information
    if add_miss_info:
        ax.set_ylabel(feature.name)
        ax.set_ylabel(ax.get_ylabel() + f" ({pct_miss_feature:0.1f}% NA)")


def plot_cate_REGR(ax,
                   feature, target,
                   feature_name=None, target_name=None,
                   target_lim=None,
                   min_width=0.2, inset_size=0.2, refline=True,
                   title=None,
                   add_miss_info=True,
                   color=list(sns.color_palette("colorblind").as_hex()),
                   verbose=True):
    """
    Plots categorical feature vs regression target (as boxplots).

    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    feature: Numpy array or Pandas series
        Categorical feature (must be of dtype object or str) to plot on y-axis.
    target: Numpy array or Pandas series
        Regression target to plot on x-axis.
    feature_name: str (optional)
        If specified used as feature's name/label in plot.
    target_name: str (optional)
        If specified used as target's name/label in plot.
    target_lim: 2-tuple of float or int
        Limits the y-axis on which the target is plotted.
    min_width: float
        Minimum width of bars. Per default the feature's member frequency determines the width per member
        which might end with a "line". 
        Can be orriden by this paramter and set to a minimum.
    inset_size: float
        Relative size of distribution bar plot on y-axis.
    refline: boolean
        Print a reference line representing the base rate.
    title: str
        Title of plot
    add_miss_info: boolean
        Show percentage of missings in feature's axis label.
    color: list or str
        Color used for bars. If list provided the second item is used.
    verbose: Boolean
        Print processing information.

    Returns
    -------
    Nothing
    """
    # Adapt feature and target
    feature, target, pct_miss_feature = _helper_adapt_feature_target(feature, target, feature_name, target_name,
                                                                     verbose=verbose)

    # Adapt color
    if isinstance(color, list):
        color = color[0]

    # Add title
    if title is None:
        title = feature.name

    # Prepare data
    df_plot = _helper_calc_barboxwidth(feature, np.tile("dummy", len(feature)),
                                       min_width=min_width)

    # Boxplot
    _ = ax.boxplot([target[feature == value] for value in df_plot[feature.name].values],
                   labels=df_plot[feature.name + "_fmt"].values,
                   widths=df_plot["w"].values,
                   vert=False,
                   patch_artist=True,
                   showmeans=True,
                   boxprops=dict(facecolor=color, alpha=0.5, color="black"),
                   medianprops=dict(color="black"),
                   meanprops=dict(marker="x",
                                  markeredgecolor="black"),
                   flierprops=dict(marker="."))
    ax.set_xlabel(target.name)
    ax.set_title(title)
    if target_lim is not None:
        ax.set_xlim(target_lim)

    # Refline
    if refline:
        ax.axvline(target.mean(), linestyle="dotted", color="black")

    # Inner barplot
    _helper_inner_barplot(ax, x=np.arange(len(df_plot)) + 1, y=df_plot["pct"],
                          inset_size=inset_size)

    # Missing information
    if add_miss_info:
        ax.set_ylabel(feature.name)
        ax.set_ylabel(ax.get_ylabel() + f" ({pct_miss_feature:0.1f}% NA)")


def plot_nume_CLASS(ax,
                    feature, target,
                    feature_name=None, target_name=None,
                    feature_lim=None,
                    inset_size=0.2, n_bins=20,
                    title=None,
                    add_miss_info=True,
                    rasterized_boxplot=False,
                    color=list(sns.color_palette("colorblind").as_hex()),
                    verbose=True):
    """
    Plots numerical feature vs classfication target (as overlayed histogram with kernel density).

    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    feature: Numpy array or Pandas series
        Numerical feature (must be of dtype float) to plot on y-axis.
    target: Numpy array or Pandas series
        Classifiction target to plot on x-axis.
    feature_name: str (optional)
        If specified used as feature's name/label in plot.
    target_name: str (optional)
        If specified used as target's name/label in plot.
    feature_lim: 2-tuple of float or int
        Limits the x-axis on which the feature is plotted.
    inset_size: float
        Relative size of distribution bar plot on y-axis.
    n_bins: int
        Number of histogram bins.
    title: str
        Title of plot
    add_miss_info: boolean
        Show percentage of missings in feature's axis label.
    rasterized_boxplot: boolean
        Should inner boxplot be rasterized. Decreases time to render plot especially for pdf.
    color: list or str
        Color used for different target categories.
    verbose: Boolean
        Print processing information.

    Returns
    -------
    Nothing
    """
    # Adapt feature and target
    feature, target, pct_miss_feature = _helper_adapt_feature_target(feature, target, feature_name, target_name,
                                                                     verbose=verbose)

    # Add title
    if title is None:
        title = feature.name

    # Adapt color
    color = color[:target.nunique()]

    # Distribution plot
    sns.histplot(ax=ax, x=feature, hue=target, hue_order=np.sort(target.unique()),
                 stat="density", common_norm=False, kde=True, bins=n_bins,
                 palette=color)
    ax.set_ylabel("Density")
    ax.set_title(title)

    # Inner Boxplot
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 1.5 * inset_size * (ylim[1] - ylim[0]))
    inset_ax = ax.inset_axes([0, 0, 1, inset_size])
    inset_ax.set_axis_off()
    ax.axhline(ylim[0], color="black")
    ax.get_shared_x_axes().join(ax, inset_ax)
    sns.boxplot(ax=inset_ax, x=feature, y=target, order=np.sort(target.unique()), orient="h", palette=color,
                showmeans=True, meanprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor": "black"})
    _ = ax.set_yticks(yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])])
    ax.set_rasterized(rasterized_boxplot)

    # Add missing information
    if add_miss_info:
        ax.set_xlabel(ax.get_xlabel() + f" ({pct_miss_feature:0.1f}% NA)")

    # Set feature_lim (must be after inner plot)
    if feature_lim is not None:
        ax.set_xlim(feature_lim)


def plot_nume_MULTICLASS(ax,
                         feature, target,
                         feature_name=None, target_name=None,
                         feature_lim=None,
                         inset_size=0.2, n_bins=20,
                         title=None,
                         add_miss_info=True,
                         rasterized_boxplot=False,
                         color=list(sns.color_palette("colorblind").as_hex()),
                         verbose=True):
    """
    Plots numerical feature vs multiclass target (as overlayed histogram with kernel density). 
    Directly calls plot_nume_CLASS with same parameter. So it is just a dummy wrapper to have consistent naming.

    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    feature: Numpy array or Pandas series
        Numerical feature to plot on y-axis.
    target: Numpy array or Pandas series
        Classifiction target to plot on x-axis.
    feature_name: str (optional)
        If specified used as feature's name/label in plot.
    target_name: str (optional)
        If specified used as target's name/label in plot.
    feature_lim: 2-tuple of float or int
        Limits the x-axis on which the feature is plotted.
    inset_size: float
        Relative size of distribution bar plot on y-axis.
    n_bins: int
        Number of histogram bins.
    title: str
        Title of plot
    add_miss_info: boolean
        Show percentage of missings in feature's axis label.
    rasterized_boxplot: boolean
        Should inner boxplot be rasterized. Decreases time to render plot especially for pdf.
    color: list or str
        Color used for different target categories.
    verbose: Boolean
        Print processing information.

    Returns
    -------
    Nothing
    """
    plot_nume_CLASS(ax=ax, feature=feature, target=target,
                    feature_name=feature_name, target_name=target_name,
                    feature_lim=feature_lim,
                    inset_size=inset_size, n_bins=n_bins,
                    title=title, add_miss_info=add_miss_info, rasterized_boxplot=rasterized_boxplot,
                    color=color,
                    verbose=verbose)


# Scatterplot as heatmap
def plot_nume_REGR(ax,
                   feature, target,
                   feature_name=None, target_name=None,
                   feature_lim=None, target_lim=None,
                   regplot=True, regplot_type="lowess", lowess_n_sample=1000, lowess_frac=2 / 3, spline_smooth=1,
                   refline=True,
                   title=None,
                   add_miss_info=True,
                   add_colorbar=True,
                   inset_size=0.2,
                   add_feature_distribution=True, add_target_distribution=True, n_bins=20,
                   add_boxplot=True, rasterized_boxplot=False,
                   colormap=LinearSegmentedColormap.from_list("bl_yl_rd", ["blue", "yellow", "red"]),
                   verbose=True):
    """
    Plots numerical feature vs regression target (as "heated" scatter plot).

    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    feature: Numpy array or Pandas series
        Numerical feature to plot on x-axis.
    target: Numpy array or Pandas series
        Regression target to plot on y-axis.
    feature_name: str (optional)
        If specified used as feature's name/label in plot.
    target_name: str (optional)
        If specified used as target's name/label in plot.
    feature_lim: 2-tuple of float or int
        Limits the x-axis on which the feature is plotted.
    target_lim: 2-tuple of float or int
        Limits the y-axis on which the target is plotted.
    regplot: boolean
        Should a regression line fitted to the scatter.
    regplot_type: str
        "lowess": Fit a non-linear (statsmodels) lowess (locally weighted scatterplot smoothing) line.
        "linear": Fit a linear regression line.
        "spline": Fit a nonlinear cubic B-spline.
    lowess_n_sample: int
        Sample data to this size before fitting lowess
    lowess_frac: float
        Between 0 and 1. "frac" parameter of statmodels lowess function.
    spline_smooth: float
        Smoothing factor of B-spline.
    refline: boolean
        Print a reference line representing the mean target value.
    title: str
        Title of plot.
    add_miss_info: boolean
        Show percentage of missings in feature's axis label.
    add_colorbar: boolean
        Show colorbar for heated dots?        
    inset_size: float
        Relative size of distribution bar plot on y-axis.
    add_feature_distribution: boolean
        Plot feature's distribution as inner histogram.
    add_target_distribution: boolean
        Plot target's distribution as inner histogram.
    n_bins: int
        Number of histogram bins in inner histogram plots.
    add_boxplot: boolean
        Show boxplot in below inner historgrams?
    rasterized_boxplot: boolean
        Should inner boxplot be rasterized. Decreases time to render plot especially for pdf.
    colormap: colormap
        Colormap used by hexbin plot.
    verbose: Boolean
        Print processing information.

    Returns
    -------
    Nothing
    """
    # Adapt feature and target
    feature, target, pct_miss_feature = _helper_adapt_feature_target(feature, target, feature_name, target_name,
                                                                     verbose=verbose)

    # Add title
    if title is None:
        title = feature.name

    '''
    # Helper for scaling of heat-points
    heat_scale = 1
    if ylim is not None:
        ax.set_ylim(ylim)
        heat_scale = heat_scale * (ylim[1] - ylim[0]) / (np.max(y) - np.min(y))
    if xlim is not None:
        ax.set_xlim(xlim)
        heat_scale = heat_scale * (xlim[1] - xlim[0]) / (np.max(feature) - np.min(feature))
    '''

    # Heatmap
    #p = ax.hexbin(feature, target, gridsize=(int(50 * heat_scale), 50), mincnt=1, cmap=color)
    p = ax.hexbin(feature, target, mincnt=1, cmap=colormap)
    ax.set_xlabel(feature.name)
    ax.set_ylabel(target.name)
    if add_colorbar:
        plt.colorbar(p, ax=ax)

    # Spline
    if regplot:
        if regplot_type == "linear":
            sns.regplot(x=feature, y=target, lowess=False, scatter=False, color="black", ax=ax)
        elif regplot_type == "spline":
            df_spline = (pd.DataFrame({"x": feature, "y": target})
                         .groupby("x")[["y"]].agg(["mean", "count"])
                         .pipe(lambda x: x.set_axis([a + "_" + b for a, b in x.columns],
                                                    axis=1, inplace=False))
                         .assign(w=lambda x: np.sqrt(x["y_count"]))
                         .sort_values("x")
                         .reset_index())
            spl = splrep(x=df_spline["x"].values, y=df_spline["y_mean"].values, w=df_spline["w"].values,
                         s=len(df_spline) * spline_smooth)
            x2 = np.quantile(df_spline["x"].values, np.arange(0.01, 1, 0.01))
            y2 = splev(x2, spl)
            ax.plot(x2, y2, color="black")

            '''
            from patsy import cr, bs
            df_spline = pd.DataFrame({"x": feature, "y": target}).sort_values("x")
            spline_basis = cr(df_spline["x"], df=7, constraints='center')
            spline_basis = bs(df_spline["x"], df=4, include_intercept=True)
            df_spline["y_spline"] = LinearRegression().fit(spline_basis, target).predict(spline_basis)
            ax.plot(df_spline["x"], df_spline["y_spline"], color="red") 
            
            sns.regplot(x=feature, y=target,
                        lowess=True,
                        scatter=False, color="green", ax=ax)
            '''
        else:
            if regplot_type != "lowess":
                warnings.warn("Wrong regplot_type, used 'lowess'")
            df_lowess = (pd.DataFrame({"x": feature, "y": target})
                         .pipe(lambda x: x.sample(min(lowess_n_sample, x.shape[0]), random_state=42))
                         .reset_index(drop=True)
                         .sort_values("x")
                         .assign(yhat=lambda x: lowess(x["y"], x["x"], frac=lowess_frac,
                                                       is_sorted=True, return_sorted=False)))
            ax.plot(df_lowess["x"], df_lowess["yhat"], color="black")

    if add_miss_info:
        ax.set_xlabel(ax.get_xlabel() + f" ({pct_miss_feature:0.1f}% NA)")

    if title is not None:
        ax.set_title(title)
    if target_lim is not None:
        ax.set_ylim(target_lim)
    if feature_lim is not None:
        ax.set_xlim(feature_lim)

    # Refline
    if refline:
        ax.axhline(target.mean(), linestyle="dashed", color="black")

    # Add y density
    if add_target_distribution:

        # Memorize ticks and limits
        xticks = ax.get_xticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create space
        ax.set_xlim(xlim[0] - 1.2 * inset_size * (xlim[1] - xlim[0]), xlim[1])

        # Create shared inset axis
        inset_ax_y = ax.inset_axes([0, 0, inset_size, 1])  # , zorder=10)
        ax.get_shared_y_axes().join(ax, inset_ax_y)

        # Plot histogram
        sns.histplot(y=target, color="grey", stat="density", kde=True, bins=n_bins, ax=inset_ax_y)
        inset_ax_y.set_ylim(ylim)

        # Remove overlayed elements
        inset_ax_y.set_xticklabels([])
        inset_ax_y.set_yticklabels([])
        inset_ax_y.set_xlabel("")
        inset_ax_y.set_ylabel("")

        # More space for plot
        left, right = inset_ax_y.get_xlim()
        inset_ax_y.set_xlim(left, right * 1.2)

        # Border
        #inset_ax_y.axvline(inset_ax_y.get_xlim()[1], color="black")

        # Remove senseless ticks
        _ = ax.set_xticks(xticks[(xticks >= xlim[0]) & (xticks <= xlim[1])])

        # Add Boxplot
        if add_boxplot:

            # Create space
            xlim_inner = inset_ax_y.get_xlim()
            inset_ax_y.set_xlim(xlim_inner[0] - 2 * inset_size * (xlim_inner[1] - xlim_inner[0]))

            # Create shared inset axis without any elements
            inset_inset_ax_y = inset_ax_y.inset_axes([0, 0, inset_size, 1])
            inset_inset_ax_y.set_axis_off()
            inset_ax_y.get_shared_y_axes().join(inset_ax_y, inset_inset_ax_y)

            # Plot boxplot
            sns.boxplot(y=target, color="lightgrey", orient="v",
                        showmeans=True, meanprops={"marker": "x",
                                                   "markerfacecolor": "white", "markeredgecolor": "white"},
                        ax=inset_inset_ax_y)
            inset_inset_ax_y.set_ylim(ylim)
            inset_inset_ax_y.set_rasterized(rasterized_boxplot)

            # More space for plot
            left, right = inset_inset_ax_y.get_xlim()
            range = right - left
            inset_inset_ax_y.set_xlim(left - range * 0.5, right)

    # Add x density
    if add_feature_distribution:

        # Memorize ticks and limits
        yticks = ax.get_yticks()
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        # Create space
        ax.set_ylim(ylim[0] - 1.2 * inset_size * (ylim[1] - ylim[0]), ylim[1])

        # Create shared inset axis
        inset_ax_x = ax.inset_axes([0, 0, 1, inset_size])  # , zorder=10)
        ax.get_shared_x_axes().join(ax, inset_ax_x)

        # Plot histogram
        sns.histplot(x=feature, color="grey", stat="density", kde=True, bins=n_bins, ax=inset_ax_x)
        inset_ax_x.set_xlim(xlim)

        # Remove overlayed elements
        inset_ax_x.set_xticklabels([])
        inset_ax_x.set_yticklabels([])
        inset_ax_x.set_xlabel("")
        inset_ax_x.set_ylabel("")

        # More space for plot
        left, right = inset_ax_x.get_ylim()
        inset_ax_x.set_ylim(left, right * 1.2)

        # Border
        #inset_ax_x.axhline(inset_ax_x.get_ylim()[1], color="black")

        # Remove senseless ticks
        _ = ax.set_yticks(yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])])

        # Add Boxplot
        if add_boxplot:

            # Create space
            ylim_inner = inset_ax_x.get_ylim()
            inset_ax_x.set_ylim(ylim_inner[0] - 2 * inset_size * (ylim_inner[1] - ylim_inner[0]))

            # Create shared inset axis without any elements
            inset_inset_ax_x = inset_ax_x.inset_axes([0, 0, 1, inset_size])
            inset_inset_ax_x.set_axis_off()
            inset_ax_x.get_shared_x_axes().join(inset_ax_x, inset_inset_ax_x)

            # PLot boxplot
            sns.boxplot(x=feature, color="lightgrey",
                        showmeans=True, meanprops={"marker": "x", "markerfacecolor": "white",
                                                   "markeredgecolor": "white"},
                        ax=inset_inset_ax_x)
            inset_inset_ax_x.set_xlim(xlim)
            inset_inset_ax_y.set_rasterized(rasterized_boxplot)

            # More space for plot
            left, right = inset_inset_ax_x.get_ylim()
            range = right - left
            inset_inset_ax_x.set_ylim(left - range * 0.5, right)

    # Hide intersection
    if add_feature_distribution and add_target_distribution:
        inset_ax_over = ax.inset_axes([0, 0, inset_size, inset_size])  # , zorder=20)
        inset_ax_over.set_facecolor("white")
        inset_ax_over.get_xaxis().set_visible(False)
        inset_ax_over.get_yaxis().set_visible(False)
        for pos in ["bottom", "left"]:
            inset_ax_over.spines[pos].set_edgecolor(None)


def plot_feature_target(ax,
                        feature, target,
                        feature_type=None, target_type=None,
                        feature_name=None, target_name=None,
                        **kwargs):
    """
    Wrapper which calls feature-target-combination's corresponding plot_nume/cate_CLASS/MULTICLASS/REGR plots
    
    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    feature: Numpy array or Pandas series
        Feature to plot.
    target: Numpy array or Pandas series
        Target to plot.
    feature_type: str
        Can be "nume" or "cate". If not specified the type is detected automatically. 
        So this automatism can be overridden.
    target_type: str
        Can be "CLASS" or "MULTICLASS" or "REGR". If not specified the type is detected automatically. 
        So this automatism can be overridden.
    feature_name: str (optional)
        If specified used as feature's name/label in plot.
    target_name: str (optional)
        If specified used as target's name/label in plot.
    **kwargs: dict
        All other possible arguments of plot_nume/cate_CLASS/MULTICLASS/REGR functions.
 
    Returns
    -------
    Nothing
    """
    # Determine feature and target type
    if feature_type is None:
        feature_type = "nume" if pd.api.types.is_numeric_dtype(feature) else "cate"
    if target_type is None:
        target_type = (dict(binary="CLASS", continuous="REGR", multiclass="MULTICLASS")
                       [type_of_target(target[~pd.Series(target).isna()])])

    # Call plot functions
    params_shared = dict(ax=ax, feature=feature, target=target,
                         feature_name=feature_name, target_name=target_name)
    if feature_type == "nume":
        if target_type == "CLASS":
            plot_nume_CLASS(**params_shared, **kwargs_reduce(kwargs, plot_nume_CLASS))
        elif target_type == "MULTICLASS":
            plot_nume_MULTICLASS(**params_shared, **kwargs_reduce(kwargs, plot_nume_MULTICLASS))
        elif target_type == "REGR":
            plot_nume_REGR(**params_shared, **kwargs_reduce(kwargs, plot_nume_REGR))
        else:
            raise Exception('Wrong TARGET_TYPE')
    else:
        if target_type == "CLASS":
            plot_cate_CLASS(**params_shared, **kwargs_reduce(kwargs, plot_nume_CLASS))
        elif target_type == "MULTICLASS":
            plot_cate_MULTICLASS(**params_shared, **kwargs_reduce(kwargs, plot_cate_MULTICLASS))
        elif target_type == "REGR":
            plot_cate_REGR(**params_shared, **kwargs_reduce(kwargs, plot_cate_REGR))
        else:
            raise Exception('Wrong TARGET_TYPE')
    # Create Frame
    # for spine in ax.spines.values():
    #    spine.set_edgecolor('black')


# Plot correlation
def plot_corr(ax, df, method, absolute=True, cutoff=None, n_jobs=1):
    """
    Correlation plot calculating pairwise correlation of all columns of "df" parameter

    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    df: Pandas dataframe
        Dataframe comprising of all features for which all pairwise correlations are calculated.
    method: str
        Can be "pearson" or "spearman" for numerical features and "cramersv" for categorical features.
    absolute: boolean
        Should the correlation values be transformed to their absolue value?
    cutoff: float   
        Plots only features with at least one (absolute) correlation value above cutoff.
    n_jobs: int
        Number of parallel processes used during of categorical correlation calculation.

    Returns
    -------
    Dataframe with correlation values.
    """

    # Check for mixed types
    count_numeric_dtypes = df.apply(lambda x: pd.api.types.is_numeric_dtype(x)).sum()
    if count_numeric_dtypes not in [0, df.shape[1]]:
        raise Exception('Mixed dtypes.')

    # Nume case
    if count_numeric_dtypes != 0:
        if method not in ["pearson", "spearman"]:
            raise Exception('False method for numeric values: Choose "pearson" or "spearman"')
        df_corr = df.corr(method=method)
        suffix = " (" + round(df.isnull().mean() * 100, 1).astype("str") + "% NA)"

    # Cate case
    else:
        if method not in ["cramersv"]:
            raise Exception('False method for categorical values: Choose "cramersv"')
        n = df.shape[1]
        df_corr = pd.DataFrame(np.zeros([n, n]), index=df.columns.values, columns=df.columns.values)
        l_tup = [(i, j) for i in range(n) for j in range(i + 1, n)]
        result = Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(corrcat)(df.iloc[:, i], df.iloc[:, j])
                                                            for i, j in l_tup)
        for k, (i, j) in enumerate(l_tup):
            df_corr.iloc[i, j] = result[k]
            #df_corr.iloc[i, j] = corrcat(df.iloc[:, i], df.iloc[:, j])
            df_corr.iloc[j, i] = df_corr.iloc[i, j]

        '''
        for i in range(n):
            for j in range(i+1, n):
                df_corr.iloc[i, j] = corrcat(df.iloc[:, i], df.iloc[:, j])
                df_corr.iloc[j, i] = df_corr.iloc[i, j]
        '''
        suffix = " (" + df.nunique().astype("str").values + ")"

    # Add info to names
    d_new_names = dict(zip(df_corr.columns.values, df_corr.columns.values + suffix))
    df_corr.rename(columns=d_new_names, index=d_new_names, inplace=True)

    # Absolute trafo
    if absolute:
        df_corr = df_corr.abs()

    # Filter out rows or cols below cutoff and then fill diagonal
    np.fill_diagonal(df_corr.values, 0)
    if cutoff is not None:
        i_cutoff = (df_corr.abs().max(axis=1) > cutoff).values
        df_corr = df_corr.loc[i_cutoff, i_cutoff]
    np.fill_diagonal(df_corr.values, 1)

    # Cluster df_corr
    tmp_order = linkage(1 - np.triu(df_corr),
                        method="average", optimal_ordering=False)[:, :2].flatten().astype(int)
    new_order = df_corr.columns.values[tmp_order[tmp_order < len(df_corr)]]
    df_corr = df_corr.loc[new_order, new_order]

    # Plot
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="Reds" if absolute else "BLues",
                xticklabels=True, yticklabels=True, ax=ax)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    ax.set_title(("Absolute " if absolute else "") + method.upper() + " Correlation" +
                 (" (cutoff: " + str(cutoff) + ")" if cutoff is not None else ""))

    return df_corr


########################################################################################################################
# Model Comparison
########################################################################################################################

# --- Non-plots --------------------------------------------------------------------------------------------------------

# Undersample
def undersample(df, target, n_max_per_level, random_state=42) -> tuple:
    """
    Undersampling dataframe with classficiation target.

    Parameters
    ----------
    df: Pandas dataframe
        Dataframe which should be undersampled.
    target: str
        Name of classification target which is the basis for undersampling.
    n_max_per_level: int
        Maximal number of obs per target category which should be returned in undersampled dataframe.
    random_state: int   
        Seed.

    Returns
    -------
    Tuple (df_under, b_sample, b_all) comprising undersampled dataframe, new undersampled base rates (tuple), 
    original base rates (tuple).
    """
    b_all = df[target].value_counts().sort_index().values / len(df)
    df_under = (df.groupby(target).apply(lambda x: x.sample(min(n_max_per_level, x.shape[0]),
                                                            random_state=random_state))
                .sample(frac=1)  # shuffle
                .reset_index(drop=True))
    b_sample = df_under[target].value_counts().sort_index().values / len(df_under)
    return df_under, b_sample, b_all


class KFoldSep(KFold):
    """
    KFold cross-validator strictly separating test-folds from all train-folds, i.e. test-folds never overlap with
    any train-fold. 
    Inherits from scikit's KFold class and only additionally provides "test_fold" parameter (boolean mask), 
    which defines test-fold. Basically works by creating default KFold splits with subsequently removing 
    non-conform indexes from train and test-folds. 

    Parameters
    ----------
    Same as for KFold
    
    Attributes
    ----------
    Same as for KFold
    """
    def __init__(self, *args, shuffle=True, **kwargs):
        super().__init__(shuffle=shuffle, *args, **kwargs)

    def split(self, X, y=None, groups=None, test_fold=None):
        i_test_fold = np.arange(len(X))[test_fold]
        for i_train, i_test in super().split(X, y, groups):
            yield i_train[~np.isin(i_train, i_test_fold)], i_test[np.isin(i_test, i_test_fold)]


class InSampleSplit:
    """
    Dummy cross-validator with test==train fold, i.e. in-sample selection, which can be used for quick change of 
    cross-validation code to non-cv.
    
    Parameters
    ----------
    shuffle: Boolean
        Shuffle data?
    random_state: int
        Seed.    
    """
    def __init__(self, shuffle=True, random_state=42):
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, *args, **kwargs):
        i_df = np.arange(len(X))
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(i_df)
        yield i_df, i_df  # train equals test

    def get_n_splits(self, *args):
        return 1


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Column selector transformer: Workaround as some scikit's ColumnTransformer versions needs same columns
    for fit and transform (bug!)

    Parameters
    ----------
    columns: list
        List of columns to seect from dataframe (input to transform)

    Attributes
    ----------
    None
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, *args):
        return self

    def transform(self, df, *args):
        return df[self.columns]



class GridSearchCV_xlgb(GridSearchCV):
    """
    GridSearchCV for XGBoost and Lightgbm models using n_estimators as incremental ("warm start") parameter.
    Inherits from scikit's GridSearchCV class (having same parameters), overwriting only the "fit" method.
    Current limitation: "scoring" parameter must be a dictionary with metric names as keys and callables a values.

    Parameters
    ----------
    Same as for GridSearchCV
    
    Attributes
    ----------
    Same as for GridSearchCV
    """
    def fit(self, X, y=None, **fit_params):
        # pdb.set_trace()

        # Adapt grid: remove n_estimators
        n_estimators = self.param_grid["n_estimators"]
        param_grid = self.param_grid.copy()
        del param_grid["n_estimators"]
        df_param_grid = pd.DataFrame(product(*param_grid.values()), columns=param_grid.keys())

        # Materialize generator as this cannot be pickled for parallel
        self.cv = list(check_cv(self.cv, y).split(X))

        # TODO: Iterate in parallel also over splits (see original fit method)
        def run_in_parallel(i):
            # for i in range(len(df_param_grid)):

            # Intialize
            df_results = pd.DataFrame()

            # Get actual parameter set
            d_param = df_param_grid.iloc[[i], :].to_dict(orient="records")[0]

            for fold, (i_train, i_test) in enumerate(self.cv):

                # pdb.set_trace()
                # Fit only once par parameter set with maximum number of n_estimators
                start = time.time()
                fit = (clone(self.estimator).set_params(**d_param,
                                                        n_estimators=int(max(n_estimators)))
                       .fit(_safe_indexing(X, i_train), _safe_indexing(y, i_train), **fit_params))
                fit_time = time.time() - start

                # Score with all n_estimators
                if hasattr(self.estimator, "subestimator"):
                    estimator = self.estimator.subestimator
                else:
                    estimator = self.estimator
                for ntree_limit in n_estimators:
                    start = time.time()
                    if isinstance(estimator, lgbm.sklearn.LGBMClassifier):
                        yhat_test = fit.predict_proba(_safe_indexing(X, i_test), num_iteration=ntree_limit)
                    elif isinstance(estimator, lgbm.sklearn.LGBMRegressor):
                        yhat_test = fit.predict(_safe_indexing(X, i_test), num_iteration=ntree_limit)
                    elif isinstance(estimator, xgb.sklearn.XGBClassifier):
                        yhat_test = fit.predict_proba(_safe_indexing(X, i_test), ntree_limit=ntree_limit)
                    else:
                        yhat_test = fit.predict(_safe_indexing(X, i_test), ntree_limit=ntree_limit)
                    score_time = time.time() - start

                    # Do it for training as well
                    if self.return_train_score:
                        if isinstance(estimator, lgbm.sklearn.LGBMClassifier):
                            yhat_train = fit.predict_proba(_safe_indexing(X, i_train), num_iteration=ntree_limit)
                        elif isinstance(estimator, lgbm.sklearn.LGBMRegressor):
                            yhat_train = fit.predict(_safe_indexing(X, i_train), num_iteration=ntree_limit)
                        elif isinstance(estimator, xgb.sklearn.XGBClassifier):
                            yhat_train = fit.predict_proba(_safe_indexing(X, i_train), ntree_limit=ntree_limit)
                        else:
                            yhat_train = fit.predict(_safe_indexing(X, i_train), ntree_limit=ntree_limit)

                    # Get time metrics
                    df_results = df_results.append(pd.DataFrame(dict(fold_type="train", fold=fold,
                                                                     scorer="time", scorer_value=fit_time,
                                                                     n_estimators=ntree_limit, **d_param),
                                                                index=[0]))
                    df_results = df_results.append(pd.DataFrame(dict(fold_type="test", fold=fold,
                                                                     scorer="time", scorer_value=score_time,
                                                                     n_estimators=ntree_limit, **d_param),
                                                                index=[0]))
                    # Get performance metrics
                    for scorer in self.scoring:
                        scorer_value = self.scoring[scorer]._score_func(_safe_indexing(y, i_test), yhat_test)
                        df_results = df_results.append(pd.DataFrame(dict(fold_type="test", fold=fold,
                                                                         scorer=scorer, scorer_value=scorer_value,
                                                                         n_estimators=ntree_limit, **d_param),
                                                                    index=[0]))
                        if self.return_train_score:
                            scorer_value = self.scoring[scorer]._score_func(_safe_indexing(y, i_train), yhat_train)
                            df_results = df_results.append(pd.DataFrame(dict(fold_type="train", fold=fold,
                                                                             scorer=scorer,
                                                                             scorer_value=scorer_value,
                                                                             n_estimators=ntree_limit, **d_param),
                                                                        index=[0]))
            return df_results

        df_results = pd.concat(Parallel(n_jobs=self.n_jobs,
                                        max_nbytes='100M')(delayed(run_in_parallel)(row)
                                                           for row in range(len(df_param_grid))))

        # Transform results
        param_names = list(np.append(df_param_grid.columns.values, "n_estimators"))
        df_cv_results = pd.pivot_table(df_results,
                                       values="scorer_value",
                                       index=param_names,
                                       columns=["fold_type", "scorer"],
                                       aggfunc=["mean", "std"],
                                       dropna=False)
        df_cv_results.columns = ['_'.join(x) for x in df_cv_results.columns.values]
        scorer_names = np.array(list(self.scoring.keys()), dtype=object)
        df_cv_results["rank_test_" + scorer_names] = df_cv_results["mean_test_" + scorer_names].rank()
        df_cv_results = df_cv_results.rename(columns={"mean_train_time": "mean_fit_time",
                                                      "mean_test_time": "mean_score_time",
                                                      "std_train_time": "std_fit_time",
                                                      "std_test_time": "std_score_time"})
        df_cv_results = df_cv_results.reset_index()
        df_cv_results["params"] = df_cv_results[param_names].apply(lambda x: x.to_dict(), axis=1)
        df_cv_results = df_cv_results.rename(columns={name: "param_" + name for name in param_names})
        self.cv_results_ = df_cv_results.to_dict(orient="list")

        # Refit
        if self.refit:
            self.scorer_ = self.scoring
            self.multimetric_ = True
            self.best_index_ = df_cv_results["mean_test_" + self.refit].idxmax()
            self.best_score_ = df_cv_results["mean_test_" + self.refit].loc[self.best_index_]
            tmp = (df_cv_results[["param_" + name for name in param_names]].loc[[self.best_index_]]
                   .to_dict(orient="records")[0])
            self.best_params_ = {key.replace("param_", ""): value for key, value in tmp.items()}
            self.best_estimator_ = (clone(self.estimator).set_params(**self.best_params_).fit(X, y, **fit_params))

        return self


# --- Plots ------------------------------------------------------------------------------------------------------------

def plot_cvresults(cv_results, metric, x_var, color_var=None, style_var=None, column_var=None, row_var=None,
                   show_gap=True, show_std=False, color=list(sns.color_palette("tab10").as_hex()),
                   height=6):
    """
    Plots results of scikit's cross-validation (i.e. a validation curve). Test performances are plotted as 
    continous lines, train performances as broken lines. Possible generalization gap lines (train minus 
    test performance) are plotted as dotted lines.
    Returns seaborn FacetGrid plot, so no ax can be specified.
    
    Parameters
    ----------
    cv_results: dict
        Result from scikit's cross-validation.
    metric: string
        Metric to plot. Occurs as "mean_train/test_<metric>" in cv_results.
    x_var: str
        Hyperparameter name to plot on x-axis (without the "param_"-prefix of cv_results).
    color_var: str
        Hyperparameter name used to group plot by color.
    style_var: str   
        Hyperparameter name used to group plot by marker style.
    column_var: str   
        Hyperparameter name used to group plot into grid columns.
    column_var: str   
        Hyperparameter name used to group plot into grid rows.
    show_gap: boolean
        Should the generalization gap be plotted on a second y-axis with a dotted line?
    show_std: boolean
        Should standard deviations of the mean performance value (aggregated over the fold runs) be plotted as bands?
    color: list
        Colors to use for color_var grouping.
    height: int
        Parameter "height" of FacetGrid.        
 
    Returns
    -------
    Seaborn FacetGrid plot.
    """
    # Transform results
    df_cvres = pd.DataFrame.from_dict(cv_results)
    df_cvres.columns = df_cvres.columns.str.replace("param_", "")
    df_cvres[x_var] = df_cvres[x_var].astype("float")
    if show_gap:
        df_cvres["gap"] = df_cvres["mean_train_" + metric] - df_cvres["mean_test_" + metric]
        gap_range = (df_cvres["gap"].min(), df_cvres["gap"].max())

    # Define plot function to use in FacetGrid
    def tmp(x, y, y2=None, std=None, std2=None, data=None,
            hue=None, style=None, color=None,
            show_gap=False, gap_range=None):

        # Test results
        sns.lineplot(x=x, y=y, data=data, hue=hue,
                     style=style, linestyle="-", markers=True if style is not None else None, dashes=False,
                     marker=True if style is not None else "o",
                     palette=color)

        # Train results
        sns.lineplot(x=x, y=y2, data=data, hue=hue,
                     style=style, linestyle="--", markers=True if style is not None else None, dashes=False,
                     marker=True if style is not None else "o",
                     palette=color)

        # Std bands
        if std is not None or std2 is not None:
            if hue is None:
                data[hue] = "dummy"
            if style is None:
                data[style] = "dummy"
            data = data.reset_index(drop=True)
            for key, val in data.groupby([hue, style]).groups.items():
                data_group = data.iloc[val, :]
                color_group = list(np.array(color)[data[hue].unique() == key[0]])[0]
                if std is not None:
                    plt.gca().fill_between(data_group[x],
                                           data_group[y] - data_group[std], data_group[y] + data_group[std],
                                           color=color_group, alpha=0.1)
                if std2 is not None:
                    plt.gca().fill_between(data_group[x],
                                           data_group[y2] - data_group[std2], data_group[y2] + data_group[std2],
                                           color=color_group, alpha=0.1)

        # Generalization gap
        if show_gap:
            ax2 = plt.gca().twinx()
            sns.lineplot(x=x, y="gap", data=data, hue=hue,
                         style=style, linestyle=":", markers=True if style is not None else None, dashes=False,
                         marker=True if style is not None else "o",
                         palette=color,
                         ax=ax2)
            ax2.set_ylabel("")
            if ax2.get_legend() is not None:
                ax2.get_legend().remove()
            ax2.set_ylim(gap_range)
            # ax2.axis("off")
        return plt.gca()

    # Plot FacetGrid
    g = (sns.FacetGrid(df_cvres, col=column_var, row=row_var, margin_titles=False if show_gap else True,
                       height=height, aspect=1)
         .map_dataframe(tmp, x=x_var, y="mean_test_" + metric, y2="mean_train_" + metric,
                        std="std_test_" + metric if show_std else None,
                        std2="std_train_" + metric if show_std else None,
                        hue=color_var, style=style_var,
                        color=color[:df_cvres[color_var].nunique()] if color_var is not None else color[0],
                        show_gap=show_gap, gap_range=gap_range if show_gap else None)
         .set_xlabels(x_var)
         .add_legend(title=None if style_var is not None else color_var))

    # Beautify and title
    g.legend._legend_box.align = "left"
    g.fig.subplots_adjust(wspace=0.2, hspace=0.1)
    g.fig.subplots_adjust(top=0.9)
    _ = g.fig.suptitle(metric + ": test (-) vs train (--)" + (" and gap (:)" if show_gap else ""), fontsize=16)
    g.tight_layout()
    return g


def plot_modelcomp(ax, df_modelcomp_result, model_var="model", run_var="run", score_var="test_score"):
    """
    Plots comparison of model performance collected in a data frame. Basically grouped boxplots overlayed by
    line plots are created.
    
    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    df_modelcomp_result: Dataframe
        Data frame comprising model comparison results.
    model_var: str
        Data frame column name denoting the model name.
    run_var: str
        Data frame column name denoting the (cross validation) run number.
    score_var: numpy array
        Data frame column name denoting the performance value of the corresponding model and run.

    Returns
    -------
    Nothing.
    """
    sns.boxplot(data=df_modelcomp_result, x=model_var, y=score_var, showmeans=True,
                meanprops={"markerfacecolor": "black", "markeredgecolor": "black"},
                ax=ax)
    sns.lineplot(data=df_modelcomp_result, x=model_var, y=score_var,
                 hue=df_modelcomp_result[run_var], linewidth=0.5, linestyle=":",
                 legend=None, ax=ax)


def plot_learningcurve(ax, n_train, score_train, score_test, time_train,
                       add_time=True,
                       color=list(sns.color_palette("tab10").as_hex())):
    """
    Plots results of scikit's learning_curve function. 
    
    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    n_train: numpy array
        "train_sizes_abs" return from scikit's learning_curve function.
    score_train: numpy array
        "train_scores" return from scikit's learning_curve function.
    score_test: numpy array
        "test_scores" return from scikit's learning_curve function.
    time_train: numpy array
        "fit_times" return from scikit's learning_curve function.
    add_time: boolean
        Plot fitting time on secondary y-axis?
    color: list
        Colors to use for train and test lines.      
 
    Returns
    -------
    Nothing.
    """
    score_train_mean = np.mean(score_train, axis=1)
    score_train_std = np.std(score_train, axis=1)
    score_test_mean = np.mean(score_test, axis=1)
    score_test_std = np.std(score_test, axis=1)
    time_train_mean = np.mean(time_train, axis=1)
    time_train_std = np.std(time_train, axis=1)

    # Plot learning curve
    ax.plot(n_train, score_train_mean, label="Train", marker='o', color=color[0])
    ax.plot(n_train, score_test_mean, label="Test", marker='o', color=color[1])
    ax.fill_between(n_train, score_train_mean - score_train_std, score_train_mean + score_train_std,
                    alpha=0.1, color=color[0])
    ax.fill_between(n_train, score_test_mean - score_test_std, score_test_mean + score_test_std,
                    alpha=0.1, color=color[1])

    # Plot fitting time
    if add_time:
        ax2 = ax.twinx()
        ax2.plot(n_train, time_train_mean, label="Time", marker="x", linestyle=":", color="grey")
        # ax2.fill_between(n_train, time_train_mean - time_train_std, time_train_mean + time_train_std,
        #                 alpha=0.1, color="grey")
        ax2.set_ylabel("Fitting time [s]")
    ax.legend(loc="best")
    ax.set_ylabel("Score")
    ax.set_title("Learning curve")


########################################################################################################################
# Interpret
########################################################################################################################

# --- Non-Plots --------------------------------------------------------------------------------------------------------

def scale_predictions(yhat, b_sample=None, b_all=None):
    """
    Rescale predictions of (multiclass) classification model (e.g. to rewind undersampling)

    Parameters
    ----------
    yhat: numpy array
        Predictions to rescale.
    b_sample: numpy array
        Base rate of undersampled data.
    b_all: numpy array
        Base rate of "orignal"" data, i.e. to which the undersampled yhat should be rescaled.

    Returns
    -------
    Numpy array with resacaled predictions
    """
    flag_1dim = False
    if b_sample is None:
        yhat_rescaled = yhat  # no rescaling case
    else:
        # Make yhat 2-dimensional (needed in classification setting)
        if yhat.ndim == 1:
            flag_1dim = True
            yhat = np.column_stack((1 - yhat, yhat))  
        yhat_unnormed = (yhat * b_all) / b_sample  # basic rescaling which also needs norming to be correct
        yhat_rescaled = yhat_unnormed / yhat_unnormed.sum(axis=1, keepdims=True)  # norming
        
        # Adapt to orignal shape
        if flag_1dim:
            yhat_rescaled = yhat_rescaled[:, 1]
            
    return yhat_rescaled


class ScalingEstimator(BaseEstimator):
    """
    Metaestimator which rescales predictions (e.g. to rewind undersampling) for any scikit classification estimator.

    Parameters
    ----------
    subestimator: scikit estimator instance(!)
        Instance of a scikit estimator, e.g. linear_model.ElasticNet(intercept=False).
    b_sample: numpy array, default=None
        Base rate of undersampled data.
    b_all: numpy array, default=None
        Base rate of "original"" data, i.e. to which the predictions of the subestimator should be rescaled.
    **kwargs: dict
        All parameters available for subestimator

    Attributes
    ----------
    classes_: array of unique target labels.
    fitted_: boolean, denoting whether instance is fitted.

    """
    def __init__(self, subestimator, b_sample=None, b_all=None, **kwargs):
        self.subestimator = subestimator
        self.b_sample = b_sample
        self.b_all = b_all
        self._estimator_type = subestimator._estimator_type
        if kwargs:
            self.subestimator.set_params(**kwargs)

    def get_params(self, deep=True):
        return dict(subestimator=self.subestimator,
                    b_sample=self.b_sample,
                    b_all=self.b_all,
                    **self.subestimator.get_params())

    def set_params(self, **params):
        if "subestimator" in params:
            self.subestimator = params["subestimator"]
            del params["subestimator"]
        if "b_sample" in params:
            self.b_sample = params["b_sample"]
            del params["b_sample"]
        if "b_all" in params:
            self.b_all = params["b_all"]
            del params["b_all"]
        self.subestimator = self.subestimator.set_params(**params)
        return self

    def fit(self, X, y, *args, **kwargs):
        self.classes_ = unique_labels(y)  # scikit requirement
        self.subestimator.fit(X, y, *args, **kwargs)
        self.fitted_ = True  # "dummy" (can be any xxx_) attribute for scikit to recognize estimator is fitted
        return self

    def predict(self, X, *args, **kwargs):
        print("")
        # Binary prediction
        if self._estimator_type == "classifier":
            yhat = self.classes_[np.argmax(self.predict_proba(X, *args, **kwargs), 
                                           axis=1)]
        # Regression
        else:
            yhat = self.subestimator.predict(X, *args, **kwargs)
        return yhat

    def predict_proba(self, X, *args, **kwargs):
        yhat = scale_predictions(self.subestimator.predict_proba(X, *args, **kwargs),
                                 self.b_sample, self.b_all)
        return yhat


class XGBClassifier_rescale(xgb.XGBClassifier):
    """
    XGBClassifier wrapper which rescales predictions (e.g. to rewind undersampling). Low level code alternative to
    ScalingEstimator, which basically only overwrites predict_proba method.

    Parameters
    ----------
    b_sample: numpy array, default=None
        Base rate of undersampled data.
    b_all: list or None, default=None
        Base rate of "original"" data, i.e. to which the predictions of the subestimator should be rescaled.
    **kwargs: dict
        All parameters available for XGBClassifier

    Attributes
    ----------
    All attributes of XGBClassifier
    """
    def __init__(self, b_sample=None, b_all=None, **kwargs):
        super().__init__(**kwargs)
        self.b_sample = b_sample
        self.b_all = b_all

    def predict_proba(self, X, *args, **kwargs):
        yhat = scale_predictions(super().predict_proba(X, *args, **kwargs),
                                 b_sample=self.b_sample, b_all=self.b_all)
        return yhat


class UndersampleEstimator(BaseEstimator):
    """
    Metaestimator which undersamples training data and rescales predictions to rewind undersampling
    for any scikit classification estimator.

    Parameters
    ----------
    subestimator: scikit estimator instance (!)
        Instance of a scikit estimator, e.g. linear_model.ElasticNet(intercept=False).
    n_max_per_level: int, default=np.inf
        Maximal number of obs per target category which should be returned in undersampled training data.    
    seed: int, default=42
        Seed.
    **kwargs: dict
        All parameters available for subestimator.

    Attributes
    ----------
    b_sample_: numpy array, comprising base rate of undersampled data.
    b_all_:  numpy array, comprising base rate of "original"" data.
    fitted_: boolean, denoting whether instance is fitted.
    """
    def __init__(self, subestimator, n_max_per_level=np.inf, seed=42, **kwargs):
        self.subestimator = subestimator
        self.n_max_per_level = n_max_per_level
        self.seed = seed  # cannot be named random_state as this might be a kwargs parameter
        self._estimator_type = subestimator._estimator_type
        if kwargs:
            self.subestimator.set_params(**kwargs)

    def get_params(self, deep=True):
        return dict(subestimator=self.subestimator,
                    n_max_per_level=self.n_max_per_level,
                    seed=self.seed,
                    **self.subestimator.get_params())

    def set_params(self, **params):
        if "subestimator" in params:
            self.subestimator = params["subestimator"]
            del params["subestimator"]
        if "n_max_per_level" in params:
            self.b_sample = params["n_max_per_level"]
            del params["n_max_per_level"]
        if "seed" in params:
            self.b_all = params["seed"]
            del params["seed"]
        self.subestimator = self.subestimator.set_params(**params)
        return self

    def fit(self, X, y, *args, **kwargs):
        if self._estimator_type == "classifier":
            self._classes = unique_labels(y)
            #self._classes = self.subestimator._classes
        # Sample and set b_sample_, b_all_
        if type_of_target(y) == "continuous":
            df_tmp = (pd.DataFrame(dict(y=y)).reset_index(drop=True).reset_index()
                      .pipe(lambda x: x.sample(min(self.n_max_per_level, x.shape[0]))))
        else:
            self.classes_ = unique_labels(y)
            df_tmp = pd.DataFrame(dict(y=y)).reset_index(drop=True).reset_index()
            self.b_all_ = df_tmp["y"].value_counts().values / len(df_tmp)
            df_tmp = (df_tmp.groupby("y")
                      .apply(lambda x: x.sample(min(self.n_max_per_level, x.shape[0]), random_state=self.seed))
                      .reset_index(drop=True))
            self.b_sample_ = df_tmp["y"].value_counts().values / len(df_tmp)
        y_under = df_tmp["y"].values
        X_under = X[df_tmp["index"].values, :]
        
        # Denote that it is fitted
        self.fitted_ = True

        # Fit
        self.subestimator.fit(X_under, y_under, *args, **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        print("")
        # Binary prediction
        if self._estimator_type == "classifier":
            yhat = self.classes_[np.argmax(self.predict_proba(X, *args, **kwargs), 
                                           axis=1)]
        # Regression
        else:
            yhat = self.subestimator.predict(X, *args, **kwargs)
        return yhat

    def predict_proba(self, X, *args, **kwargs):
        yhat = scale_predictions(self.subestimator.predict_proba(X, *args, **kwargs),
                                 self.b_sample_, self.b_all_)
        return yhat


class LogtrafoEstimator(BaseEstimator):
    """
    Metaestimator for any scikit regression estimator which log-transforms target in training labels 
    and retransforms predictions to rewind, including variance estimation or residuals in order 
    to adapt calculation of expected value.

    Parameters
    ----------
    subestimator: scikit estimator instance (!)
        Instance of a scikit estimator, e.g. linear_model.ElasticNet(intercept=False).
    variance_scaling_factor: float, default=1
        Factor to be multiplied on varinace estimation in order to adapt predicitons to adapt calibration of
        predictions.
    **kwargs: dict
        All parameters available for subestimator.

    Attributes
    ----------
    varest_: float, variance of residuals.
    fitted_: boolean, denoting whether instance is fitted.
    """
    def __init__(self, subestimator, variance_scaling_factor=1, **kwargs):
        self.subestimator = subestimator
        self.variance_scaling_factor = variance_scaling_factor
        self._estimator_type = subestimator._estimator_type
        if kwargs:
            self.subestimator.set_params(**kwargs)

    def get_params(self, deep=True):
        return dict(subestimator=self.subestimator,
                    variance_scaling_factor=self.variance_scaling_factor,
                    **self.subestimator.get_params())

    def set_params(self, **params):
        if "subestimator" in params:
            self.subestimator = params["subestimator"]
            del params["subestimator"]
        if "variance_scaling_factor" in params:
            self.b_sample = params["variance_scaling_factor"]
            del params["variance_scaling_factor"]
        self.subestimator = self.subestimator.set_params(**params)
        return self

    def fit(self, X, y, *args, **kwargs):
        self.subestimator.fit(X, np.log(1 + y), *args, **kwargs)
        
        # Calculate an overall error variance by residuals
        res = self.subestimator.predict(X) - np.log(1 + y)
        print(np.std(res)**2)
        self.varest_ = np.var(res)
        self.fitted_ = True
        return self

    def predict(self, X, *args, **kwargs):
        # Retransform respecting error variance 
        yhat = (np.exp(self.subestimator.predict(X, *args, **kwargs) +
                       0.5 * self.variance_scaling_factor * self.varest_) - 1)
        return yhat


def varimp2df(varimp, features):
    """ 
    Convert result of scikit's variable importance to a dataframe with additional information regarding 
    variable importance.
    
    Parameters
    ----------
    varimp: dict
        Feature importance as returned from scikit's permutation_importance function comprinsing at least 
        "importances_mean" key
    features: list
        List of features

    Returns
    -------
    Dataframe with additional computations regarding variable importace
    """
    df_varimp = (pd.DataFrame(dict(score_diff=varimp["importances_mean"], feature=features))
                 .sort_values(["score_diff"], ascending=False).reset_index(drop=True)
                 .assign(importance=lambda x: 100 * np.where(x["score_diff"] > 0,
                                                             x["score_diff"] / max(x["score_diff"]), 0),
                         importance_cum=lambda x: 100 * x["importance"].cumsum() / sum(x["importance"])))
    return df_varimp


def variable_importance(estimator, df, y, features, scorer, target_type=None,
                        n_jobs=None, random_state=None):
    """ 
    Dataframe based permutation importance which can select a subset of features for which to calculate VI
    (in contrast to scikit's permutation_importance function).
    
    Parameters
    ----------
    estimator: scikit estimator
        Fitted estimator for which to calculate variable importance.
    df: Pandas dataframe
        Dataframe for which to calculate importance, usually training or test data.
    y: Numpy array or Pandas series
        Target variable, needed to calcuate score
    features: list
        List of features for which to calculate importance.
    scorer: sklearn.metrics scorer 
        Metric which is the basis to calcuate importance
    target_type: str
        Can be "CLASS" or "MULTICLASS" or "REGR". If not specified the type is detected automatically from y. 
        So this automatism can be overridden.
    n_jobs: int
        Number of parallel processes used. Each feature importance is calculated in parallel by its own 
        process.
    random_state: int
        Seed (used for permutation). Same for each feature.

    Returns
    -------
    Dataframe with additional computations regarding variable importance.
    """
    # Original performance
    if target_type is None:
        target_type = dict(continuous="REGR", binary="CLASS", multiclass="MULTICLASS")[type_of_target(y)]
    yhat = estimator.predict(df) if target_type == "REGR" else estimator.predict_proba(df)
    score_orig = scorer._score_func(y, yhat)

    # Performance per variable after permutation
    np.random.seed(random_state)
    i_perm = np.random.permutation(np.arange(len(df)))  # permutation vector, so permute each feature the same way

    def run_in_parallel(df, feature):
        df_copy = df.copy()
        df_copy[feature] = df_copy[feature].values[i_perm]  # permute
        yhat_perm = estimator.predict(df_copy) if target_type == "REGR" else estimator.predict_proba(df_copy)
        score = scorer._score_func(y, yhat_perm)
        return score
    scores = Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(run_in_parallel)(df, feature)
                                                        for feature in features)
    
    # Calc performance diff, i.e. importance
    df_varimp = varimp2df({"importances_mean": score_orig - scores}, features)
    return df_varimp


def partial_dependence(estimator, df, features,
                       df_ref=None, quantiles=np.arange(0.05, 1, 0.1),
                       n_jobs=4):
    """ 
    Dataframe based patial dependence which can use a reference dataset for value-grid defintion
    (in contrast to scikit's partial_dependence function).
    
    Parameters
    ----------
    estimator: sklearn estimator
        Fitted estimator for which to calculate partial dependence.
    df: Pandas dataframe
        Dataframe for which to calculate partial dependence, usually training or test data.
    features: list
        List of features for which to calculate partial dependence.
    df_ref: Pandas dataframe, default=df
        Reference dataframe which is used for value-grid definition.  
    quantiles: numpy array
        Array of quantiles to use for numeric features as grid  
    n_jobs: int
        Number of parallel processes used. Each feature's partial dependence is calculated in parallel by its own 
        process.

    Returns
    -------
    Dataframe with patial depdence information
    """
    # Set reference data
    if df_ref is None:
        df_ref = df

    # Calc partial dependence
    def run_in_parallel(feature, df, df_ref):
        
        # Derive value grid for which dependence is calculated
        if pd.api.types.is_numeric_dtype(df_ref[feature]):
            values = np.unique(df_ref[feature].quantile(quantiles).values)
        else:
            values = df_ref[feature].unique()

        # Loop over value grid and calc dependence
        df_copy = df.copy()  # save original data
        df_pd_feature = pd.DataFrame()
        for value in values:
            df_copy[feature] = value
            df_pd_feature = df_pd_feature.append(
                pd.DataFrame(np.mean(estimator.predict_proba(df_copy) if estimator._estimator_type == "classifier" else
                                     estimator.predict(df_copy), axis=0).reshape(1, -1)))
        df_pd_feature.columns = ["yhat"] if estimator._estimator_type == "regressor" else estimator.classes_
        df_pd_feature["value"] = values
        df_pd_feature = df_pd_feature.reset_index(drop=True)

        return df_pd_feature

    # Run in parallel and append
    l_pd = (Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(run_in_parallel)(feature, df, df_ref)
                                                       for feature in features))
    d_pd = dict(zip(features, l_pd))
    return d_pd


def shap2pd(shap_values, features,
            df_ref=None, n_bins=10, format_string=".2f"):
    """ 
    Aggregate shapley to partial dependence.
    
    Parameters
    ----------
    shap_values: dict
        Shap values as returned from any SHAP explainer.
    features: list
        List of features for which to calculate partial dependence.
    df_ref: Pandas dataframe, default=shap_values.data
        Reference dataframe which is used for value-grid definition.  
    n_bins: int
        Number of bins to use for quantile based binning of numeric feature, used as grouping for shap aggregation.
    format_string: str
        Format to use for bin edges of numeric feature binning.

    Returns
    -------
    Dataframe with patial depdence information
    """
    if df_ref is None:
        df_ref = pd.DataFrame(shap_values.data, columns=shap_values.feature_names[0])

    d_pd = dict()
    for feature in features:
        # Location of feature in shap_values
        i_feature = np.argwhere(shap_values.feature_names[0] == feature)[0][0]
        intercept = shap_values.base_values[0]

        # Numeric features: Create bins
        if pd.api.types.is_numeric_dtype(df_ref[feature]):
            kbinsdiscretizer_fit = KBinsDiscretizer(n_bins=n_bins, encode="ordinal").fit(df_ref[[feature]])
            bin_edges = kbinsdiscretizer_fit.bin_edges_
            bin_labels = np.array(["q" + format(i + 1, "02d") + " (" + 
                                   format(bin_edges[0][i], format_string) + " - " +
                                   format(bin_edges[0][i + 1], format_string) + ")"
                                   for i in range(len(bin_edges[0]) - 1)])
            df_shap = pd.DataFrame({"value": bin_labels[(kbinsdiscretizer_fit
                                                         .transform(shap_values.data[:, [i_feature]])[:, 0])
                                                        .astype(int)],
                                    "yhat": shap_values.values[:, i_feature]})  # TODO: MULTICLASS

        # Categorical feature
        else:
            df_shap = pd.DataFrame({"value": shap_values.data[:, i_feature],
                                    "yhat": shap_values.values[:, i_feature]})  # TODO: MULTICLASS

        # Aggregate and add intercept
        df_shap_agg = (df_shap.groupby("value").mean().reset_index()
                       .assign(yhat=lambda x: x["yhat"] + intercept))
        d_pd[feature] = df_shap_agg
    return d_pd


# TODO: remove len_nume paramter and make it flexible
def agg_shap_values(shap_values, df_explain, len_nume, l_map_onehot, round=2):
    """
    # Aggregate onehot encoded shapely values
        
    Parameters
    ----------
    shap_values: dict
        Shap values as returned from any SHAP explainer.
    df_explain: Pandas dataframe 
        Dataframe used to create matrix which is send to shap explainer.
    len_nume: int
        Number of numerical features building first columns of df_explain.
    l_map_onehot: list
        Attribute categories_ of used onehot-encoder (or similar list).
    round: int
        Rounding shap values
        
    Returns
    -------
    Aggregated shap values (same structure as shap_values parameter)
    """

    # Copy
    shap_values_agg = copy.copy(shap_values)

    # Adapt feature_names
    shap_values_agg.feature_names = np.tile(df_explain.columns.values, (len(shap_values_agg), 1))

    # Adapt display data
    shap_values_agg.data = df_explain.round(round).values

    # Care for multiclass
    values_3d = np.atleast_3d(shap_values_agg.values)
    a_shap = np.empty((values_3d.shape[0], df_explain.shape[1], values_3d.shape[2]))
    # for k in range(a_shap.shape[2]):

    # Initilaize with nume shap valus (MUST BE AT BEGINNING OF df_explain): TODO: remove and make flexible
    start_cate = len_nume
    a_shap[:, 0:start_cate, :] = values_3d[:, 0:start_cate, :].copy()

    # Aggregate cate shap values
    for i in range(len(l_map_onehot)):
        step = len(l_map_onehot[i])
        a_shap[:, len_nume + i, :] = values_3d[:, start_cate:(start_cate + step), :].sum(axis=1)
        start_cate = start_cate + step

    # Adapt non-multiclass
    if a_shap.shape[2] == 1:
        a_shap = a_shap[:, :, 0]
    shap_values_agg.values = a_shap

    # Return
    return shap_values_agg


# --- Plots ------------------------------------------------------------------------------------------------------------

def plot_roc(ax, y, yhat, annotate=True, fontsize=10,
             color=list(sns.color_palette("colorblind").as_hex()), 
             target_labels=None):
    """
    Plot roc curve. Also capable of Multiclass data, plotting OvR (one-vs-rest) roc curves. Additionally capable
    of Regression data, allowing concordance interpretation.
        
    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    annotate: boolean
        Annotate curve with threshold values.
    fontsize: int
        Fontsize of annotations.
    color: list
        Colors used for Multiclass roc curves.
    target_labels: list
        Labels used for Multiclass roc curves.
        
    Returns
    -------
    Nothing
    """

    # Also for regression: squeeze between 0 and 1
    if (y.ndim == 1) & (yhat.ndim == 1):
        if (np.min(y) < 0) | (np.max(y) > 1):
            y = MinMaxScaler().fit_transform(y.reshape(-1, 1))[:, 0]
        if (np.min(yhat) < 0) | (np.max(yhat) > 1):
            yhat = MinMaxScaler().fit_transform(yhat.reshape(-1, 1))[:, 0]

    # CLASS (and regression)
    if yhat.ndim == 1:
        
        # Roc curve
        fpr, tpr, cutoff = roc_curve(y, yhat)
        cutoff[0] = 1
        roc_auc = roc_auc_score(y, yhat)
        ax.plot(fpr, tpr)  # sns.lineplot(fpr, tpr, ax=ax, palette=sns.xkcd_palette(["red"]))        
        ax.set_title(f"ROC (AUC = {roc_auc:0.2f})")
        
        # Annotate text
        if annotate:
            for thres in np.arange(0.1, 1, 0.1):
                i_thres = np.argmax(cutoff < thres)
                ax.annotate(f"{thres:0.1f}", (fpr[i_thres], tpr[i_thres]), fontsize=fontsize)

    # MULTICLASS
    else:
        
        # OvR (one-vs-rest) auc calculation
        n_classes = yhat.shape[1]
        aucs = np.array([round(roc_auc_score(np.where(y == i, 1, 0), 
                                             yhat[:, i]), 2) for i in np.arange(n_classes)])  
        
        # Roc curves
        for i in np.arange(n_classes):
            y_bin = np.where(y == i, 1, 0)
            fpr, tpr, _ = roc_curve(y_bin, yhat[:, i])
            if target_labels is not None:
                new_label = str(target_labels[i]) + " (" + str(aucs[i]) + ")"
            else:
                new_label = str(i) + " (" + str(aucs[i]) + ")"
            ax.plot(fpr, tpr, color=color[i], label=new_label)
            
        # Title and legend
        mean_auc = np.average(aucs).round(3)
        weighted_auc = np.average(aucs, weights=np.array(np.unique(y, return_counts=True))[1, :]).round(3)
        ax.set_title("ROC\n" + r"($AUC_{mean}$ = " + str(mean_auc) + r", $AUC_{weighted}$ = " +
                     str(weighted_auc) + ")")
        ax.legend(title=r"Target ($AUC_{OvR}$)", loc='best')

    # Axis labels
    ax.set_xlabel(r"fpr: P($\^y$=1|$y$=0)")
    ax.set_ylabel(r"tpr: P($\^y$=1|$y$=1)")


def plot_calibration(ax, y, yhat, n_bins=5,
                     color=list(sns.color_palette("colorblind").as_hex()), 
                     target_labels=None):
    """
    Plot calibration curve. 
        
    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    n_bins: int
        Number of bins in calibration plot.
    color: list
        Colors used for Multiclass calibration curves.
    target_labels: list
        Labels used for Multiclass calibration curves.
        
    Returns
    -------
    Nothing
    """
    
    # Initialize settings needed for diagonal line processing
    minmin = np.inf
    maxmax = -np.inf
    max_yhat = -np.inf

    # Plot
    if yhat.ndim > 1:
        n_classes = yhat.shape[1]
    else:
        n_classes = 1
    for i in np.arange(n_classes):
        df_plot = (pd.DataFrame({"y": np.where(y == i, 1, 0) if yhat.ndim > 1 else y,
                                 "yhat": yhat[:, i] if yhat.ndim > 1 else yhat})
                   .assign(bin=lambda x: pd.qcut(x["yhat"], n_bins, duplicates="drop").astype("str"))
                   .groupby(["bin"], as_index=False).agg("mean")
                   .sort_values("yhat"))
        ax.plot(df_plot["yhat"], df_plot["y"], "o-", color=color[i],
                label=target_labels[i] if target_labels is not None else str(i))

        # Get limits
        minmin = min(minmin, min(df_plot["y"].min(), df_plot["yhat"].min()))
        maxmax = max(maxmax, max(df_plot["y"].max(), df_plot["yhat"].max()))
        max_yhat = max(max_yhat, df_plot["yhat"].max())

    # Diagonal line
    ax.plot([minmin, maxmax], [minmin, maxmax], linestyle="--", color="grey")

    # Focus
    ax.set_xlim(None, maxmax + 0.05 * (maxmax - minmin))
    ax.set_ylim(None, maxmax + 0.05 * (maxmax - minmin))

    # Labels and legend
    props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
             'ylabel': r"$\bar{y}$ in $\^y$-bin",
             'title': "Calibration"}
    _ = ax.set(**props)
    if yhat.ndim > 1:
        ax.legend(title="Target", loc='best')


def plot_confusion(ax, y, yhat, threshold=0.5, cmap="Blues", target_labels=None):
    """
    Plot confusion matrix. 
        
    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    threshold: float
        Predictions above threshold are treated as "positive" class.
    cmap: str
        Cmap of heatmap plot.
    target_labels: list
        Descriptive labels used for axis.
        
    Returns
    -------
    Nothing
    """
    # Binary label
    if yhat.ndim == 1:
        yhat_bin = np.where(yhat > threshold, 1, 0)
    else:
        yhat_bin = yhat.argmax(axis=1)

    # Confusion dataframe
    unique_y = np.unique(y)
    freq_y = np.unique(y, return_counts=True)[1]
    freqpct_y = np.round(np.divide(freq_y, len(y)) * 100, 1)
    freq_yhat = np.unique(np.concatenate((yhat_bin, unique_y)), return_counts=True)[1] - 1
    freqpct_yhat = np.round(np.divide(freq_yhat, len(y)) * 100, 1)
    m_conf = confusion_matrix(y, yhat_bin)
    if target_labels is None:
        target_labels = unique_y
    ylabels = [str(target_labels[i]) + " (" + str(freq_y[i]) + ": " + str(freqpct_y[i]) + "%)" for i in
               np.arange(len(target_labels))]
    xlabels = [str(target_labels[i]) + " (" + str(freq_yhat[i]) + ": " + str(freqpct_yhat[i]) + "%)" for i in
               np.arange(len(target_labels))]
    df_conf = (pd.DataFrame(m_conf, columns=target_labels, index=target_labels)
               .rename_axis(index="True label",
                            columns="Predicted label"))

    # accuracy and confusion calculation
    acc = accuracy_score(y, yhat_bin)

    # "Plot" dataframe as heatmap
    sns.heatmap(df_conf, annot=True, fmt=".5g", cmap=cmap, ax=ax,
                xticklabels=True, yticklabels=True, cbar=False)
    ax.set_yticklabels(labels=ylabels, rotation=0)
    ax.set_xticklabels(labels=xlabels, 
                       rotation=45 if yhat.ndim > 1 else 0,
                       ha="right" if yhat.ndim > 1 else "center")
    ax.set_xlabel("Predicted label (#: %)")
    ax.set_ylabel("True label (#: %)")
    ax.set_title("Confusion Matrix ($Acc_{" +
                 (format(threshold, "0.2f") if yhat.ndim == 1 else "") +
                 "}$ = " + format(acc, "0.2f") + ")")
    for text in ax.texts[::len(target_labels) + 1]:
        text.set_weight('bold')


def plot_confusionbars(ax, y, yhat, y_as_category=True, target_labels=None):
    """
    Plot confusion bars. 
        
    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    y_as_category: boolean
        If True use y (true labels) as categories. Otherwise use yhat.
    target_labels: list
        Descriptive labels used for axis.
        
    Returns
    -------
    Nothing
    """
    n_classes = yhat.shape[1]

    # Make y and yhat series
    y = pd.Series(y, name="y").astype(str)
    yhat = pd.Series(yhat.argmax(axis=1), name="yhat").astype(str)

    # Map labels
    if target_labels is not None:
        d_map = {str(i): str(target_labels[i]) for i in np.arange(n_classes)}
        y = y.map(d_map)
        yhat = yhat.map(d_map)

    # Plot and adapt
    if y_as_category:
        plot_cate_MULTICLASS(ax, feature=y, target=yhat, reverse=True)
    else:
        plot_cate_MULTICLASS(ax, feature=yhat, target=y, exchange_x_y_axis=True)
    ax.set_xlabel("% Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("")
    ax.get_legend().remove()


def plot_multiclass_metrics(ax, y, yhat, target_labels=None):
    """
    Plot multiclass metrics table. 
        
    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    target_labels: list
        Descriptive labels used for axis.
        
    Returns
    -------
    Nothing
    """
    
    # Calculate metrics
    m_conf = confusion_matrix(y, yhat.argmax(axis=1))
    aucs = np.array([round(roc_auc_score(np.where(y == i, 1, 0), yhat[:, i]), 2) for i in np.arange(yhat.shape[1])])
    prec = np.round(np.diag(m_conf) / m_conf.sum(axis=0) * 100, 1)
    rec = np.round(np.diag(m_conf) / m_conf.sum(axis=1) * 100, 1)
    f1 = np.round(2 * prec * rec / (prec + rec), 1)

    # Add top3 metrics
    if target_labels is None:
        target_labels = np.unique(y).tolist()
    df_metrics = (pd.DataFrame(np.column_stack((y, np.flip(np.argsort(yhat, axis=1), axis=1)[:, :3])),
                               columns=["y", "yhat1", "yhat2", "yhat3"])
                  .assign(acc_top1=lambda x: (x["y"] == x["yhat1"]).astype("int"),
                          acc_top2=lambda x: ((x["y"] == x["yhat1"]) | (x["y"] == x["yhat2"])).astype("int"),
                          acc_top3=lambda x: ((x["y"] == x["yhat1"]) | (x["y"] == x["yhat2"]) |
                                              (x["y"] == x["yhat3"])).astype("int"))
                  .assign(label=lambda x: np.array(target_labels, dtype="object")[x["y"].values])
                  .groupby(["label"])["acc_top1", "acc_top2", "acc_top3"].agg("mean").round(2)
                  .join(pd.DataFrame(np.stack((aucs, rec, prec, f1), axis=1),
                                     index=target_labels, columns=["auc", "recall", "precision", "f1"])))
    
    # "Plot" df_metrics as heatmap without color
    sns.heatmap(df_metrics.T, annot=True, fmt=".5g",
                cmap=ListedColormap(['white']), linewidths=2, linecolor="black", cbar=False,
                ax=ax, xticklabels=True, yticklabels=True)
    ax.set_yticklabels(labels=['Accuracy\n Top1', 'Accuracy\n Top2', 'Accuracy\n Top3', "AUC\n 1-vs-all",
                               'Recall\n' r"P($\^y$=k|$y$=k))", 'Precision\n' r"P($y$=k|$\^y$=k))", 'F1'])
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(left=False, top=False)
    ax.set_xlabel("True label")


def plot_precision_recall(ax, y, yhat, annotate=True, fontsize=10):
    """
    Plot precision recall curve. 
        
    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    annotate: boolean
        Annotate curve with precision recall values.
    fontsize: int
        Fontsize of annotations.
        
    Returns
    -------
    Nothing
    """

    # Calculate precision recall
    prec, rec, cutoff = precision_recall_curve(y, yhat)
    cutoff = np.append(cutoff, 1)
    prec_rec_auc = average_precision_score(y, yhat)

    # Plot curve
    ax.plot(rec, prec)
    props = {'xlabel': r"recall=tpr: P($\^y$=1|$y$=1)",
             'ylabel': r"precision: P($y$=1|$\^y$=1)",
             'title': f"Precision Recall Curve (AUC = {prec_rec_auc:0.2f})"}
    ax.set(**props)

    # Annotate text
    if annotate:
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            ax.annotate(f"{thres:0.1f}", (rec[i_thres], prec[i_thres]), fontsize=fontsize)


def plot_precision(ax, y, yhat, annotate=True, fontsize=10):
    """
    Plot precision curve. 
        
    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    annotate: boolean
        Annotate curve with precision values.
    fontsize: int
        Fontsize of annotations.
        
    Returns
    -------
    Nothing
    """

    # Calculate precision and pct_tested (percentage tested)
    pct_tested = np.array([])
    prec, _, cutoff = precision_recall_curve(y, yhat)
    cutoff = np.append(cutoff, 1)
    for thres in cutoff:
        pct_tested = np.append(pct_tested, [np.sum(yhat >= thres) / len(yhat)])

    # Plot curve    
    ax.plot(pct_tested, prec)  #sns.lineplot(pct_tested, prec[:-1], ax=ax, palette=sns.xkcd_palette(["red"]))
    props = {'xlabel': "% Samples Tested",
             'ylabel': r"precision: P($y$=1|$\^y$=1)",
             'title': "Precision Curve"}
    ax.set(**props)

    # Annotate text
    if annotate:
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            if i_thres:
                ax.annotate(f"{thres:0.1f}", (pct_tested[i_thres], prec[i_thres]),
                            fontsize=fontsize)


# Plot model performance for CLASS target
def get_plotcalls_model_performance_CLASS(y, yhat,
                                          n_bins=5, threshold=0.5, cmap="Blues", annotate=True, fontsize=10):
    """
    Get dictionary of plot calls for predictive performance of classification. 
        
    Parameters
    ----------
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    n_bins: int
        Number of bins in calibration plot.
    threshold: float
        Predictions above threshold are treated as "positive" class.
    cmap: str
        Cmap of heatmap plot.
    annotate: boolean
        Annotate curves with threshold values.
    fontsize: int
        Fontsize of annotations.
        
    Returns
    -------
    Dictionary with plot calls
    """
    # Convert yhat to 1-dim
    if ((yhat.ndim == 2) and (yhat.shape[1] == 2)):
        yhat = yhat[:, 1]

    # Define plot dict
    d_calls = dict()
    d_calls["roc"] = (plot_roc, dict(y=y, yhat=yhat, annotate=annotate, fontsize=fontsize))
    d_calls["confusion"] = (plot_confusion, dict(y=y, yhat=yhat, threshold=threshold, cmap=cmap))
    d_calls["distribution"] = (plot_nume_CLASS, dict(feature=yhat, target=y, feature_lim=(0, 1),
                                                     feature_name=r"Predictions ($\^y$)",
                                                     add_miss_info=False))
    d_calls["calibration"] = (plot_calibration, dict(y=y, yhat=yhat, n_bins=n_bins))
    d_calls["precision_recall"] = (plot_precision_recall, dict(y=y, yhat=yhat, annotate=annotate, fontsize=fontsize))
    d_calls["precision"] = (plot_precision, dict(y=y, yhat=yhat, annotate=annotate, fontsize=fontsize))

    return d_calls


# Plot model performance for CLASS target
def get_plotcalls_model_performance_MULTICLASS(y, yhat,
                                               n_bins=5, cmap="Blues", annotate=True, fontsize=10,
                                               target_labels=None):
    """
    Get dictionary of plot calls for predictive performance of multiclass classification. 
        
    Parameters
    ----------
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    n_bins: int
        Number of bins in calibration plot.
    cmap: str
        Cmap of heatmap plot.
    annotate: boolean
        Annotate curves with precision recall values.
    fontsize: int
        Fontsize of annotations.
    target_labels: list
        Descriptive labels used for axis.
        
    Returns
    -------
    Dictionary with plot calls
    """
    # Define plot dict
    d_calls = dict()
    d_calls["roc"] = (plot_roc, dict(y=y, yhat=yhat, target_labels=target_labels))
    d_calls["confusion"] = (plot_confusion, dict(y=y, yhat=yhat, threshold=None, cmap=cmap,
                                                 target_labels=target_labels))
    d_calls["true_bars"] = (plot_confusionbars, dict(y=y, yhat=yhat, y_as_category=True, target_labels=target_labels))
    d_calls["calibration"] = (plot_calibration, dict(y=y, yhat=yhat, n_bins=n_bins, target_labels=target_labels))
    d_calls["pred_bars"] = (plot_confusionbars, dict(y=y, yhat=yhat, y_as_category=False, target_labels=target_labels))
    d_calls["multiclass_metrics"] = (plot_multiclass_metrics, dict(y=y, yhat=yhat, target_labels=target_labels))

    return d_calls


# Plot model performance for CLASS target
def get_plotcalls_model_performance_REGR(y, yhat,
                                         ylim, regplot, n_bins):
    """
    Get dictionary of plot calls for predictive performance of regression. 
        
    Parameters
    ----------
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    ylim: tuple: (float, float)
        Limit of y in "observeds_vs_fitted" plot and of yhat in residual plots.
    regplot: boolean
        Should a regression line fitted to the scatter.
    n_bins: int
        Number of bins in calibration plot.
        
    Returns
    -------
    Dictionary with plot calls
    """
    # Convert yhat to 1-dim
    if ((yhat.ndim == 2) and (yhat.shape[1] == 2)):
        yhat = yhat[:, 1]

    # Define plot dict
    d_calls = dict()
    title = r"Observed vs. Fitted ($\rho_{Spearman}$ = " + f"{spear(y, yhat):0.2f})"
    d_calls["observed_vs_fitted"] = (plot_nume_REGR, dict(feature=yhat, target=y,
                                                          feature_name=r"$\^y$", target_name="y",
                                                          title=title,
                                                          feature_lim=ylim,
                                                          regplot=regplot,
                                                          add_miss_info=False))
    d_calls["calibration"] = (plot_calibration, dict(y=y, yhat=yhat, n_bins=n_bins))
    d_calls["distribution"] = (plot_nume_CLASS, dict(feature=np.append(y, yhat),
                                                     target=np.append(np.tile("y", len(y)),
                                                                      np.tile(r"$\^y$", len(yhat))),
                                                     feature_name="",
                                                     target_name="",
                                                     title="Distribution",
                                                     add_miss_info=False,))
    d_calls["residuals_vs_fitted"] = (plot_nume_REGR, dict(feature=yhat, target=y - yhat,
                                                           feature_name=r"$\^y$",
                                                           target_name=r"y-$\^y$",
                                                           title="Residuals vs. Fitted",
                                                           feature_lim=ylim,
                                                           regplot=regplot,
                                                           add_miss_info=False))

    d_calls["absolute_residuals_vs_fitted"] = (plot_nume_REGR, dict(feature=yhat, target=abs(y - yhat),
                                                                    feature_name=r"$\^y$",
                                                                    target_name=r"|y-$\^y$|",
                                                                    title="Absolute Residuals vs. Fitted",
                                                                    feature_lim=ylim,
                                                                    regplot=regplot,
                                                                    add_miss_info=False))

    d_calls["relative_residuals_vs_fitted"] = (plot_nume_REGR, dict(feature=yhat,
                                                                    target=np.where(y == 0, np.nan,
                                                                                    abs(y - yhat) / abs(y)),
                                                                    feature_name=r"$\^y$",
                                                                    target_name=r"|y-$\^y$|/|y|",
                                                                    title="Relative Residuals vs. Fitted",
                                                                    feature_lim=ylim,
                                                                    regplot=regplot,
                                                                    add_miss_info=False))

    return d_calls


def get_plotcalls_model_performance(y, yhat, target_type=None,
                                    n_bins=5, threshold=0.5, target_labels=None,
                                    cmap="Blues", annotate=True, fontsize=10,
                                    ylim=None, regplot=True,
                                    l_plots=None):
    """
    Get dictionary of plot calls for predictive performance.
    Wrapper for plot_model_performance_<target_type> 
        
    Parameters
    ----------
    y: Numpy array or Pandas series
        Target.
    yhat: Numpy array or Pandas series
        Predictions.
    target_type: str
        Can be "CLASS" or "MULTICLASS" or "REGR". If not specified the type is detected automatically. 
        So this automatism can be overridden.
    n_bins: int
        Number of bins in calibration plot.
    threshold: float, only used for CLASS
        Predictions above threshold are treated as "positive" class.
    target_labels: list, only used for MULTICLASS
        Descriptive labels used for axis.
    cmap: str, only used for CLASS and MULTICLASS
        Cmap of heatmap plot.        
    annotate: boolean, only used for CLASS and MULTICLASS
        Annotate curves with precision recall values.
    fontsize: int, only used for CLASS and MULTICLASS
        Fontsize of annotations.        
    ylim: tuple: (float, float), only used for REGR
        Limit of y in "observeds_vs_fitted" plot and of yhat in residual plots.        
    regplot: boolean, only used for REGR
        Should a regression line fitted to the scatter.    
    l_plots: list
        Names of plots which should be added.
        Default:  
        CLASS: "roc", "confusion", "distribution", "calibration", "precision_recall", "precision"
        MULTICLASS: "roc", "confusion", "true_bars", "calibration", "pred_bars", "multiclass_metrics"
        REGR: "observed_vs_fitted", "calibration", "distribution", 
                  "residuals_vs_fitted", "absolute_residuals_vs_fitted", "relative_residuals_vs_fitted"        

    Returns
    -------
    Dictionary with plot calls which can be used by plot_l_calls.
    """
    # Derive target type
    if target_type is None:
        target_type = dict(continuous="REGR", binary="CLASS", multiclass="MULTICLASS")[type_of_target(y)]

    # Plot
    if target_type == "CLASS":
        d_calls = get_plotcalls_model_performance_CLASS(
            y=y, yhat=yhat, n_bins=n_bins, threshold=threshold, cmap=cmap, annotate=annotate, fontsize=fontsize)
    elif target_type == "REGR":
        d_calls = get_plotcalls_model_performance_REGR(
            y=y, yhat=yhat, ylim=ylim, regplot=regplot, n_bins=n_bins)
    elif target_type == "MULTICLASS":
        d_calls = get_plotcalls_model_performance_MULTICLASS(
            y=y, yhat=yhat, n_bins=n_bins, target_labels=target_labels, cmap=cmap, annotate=annotate, fontsize=fontsize)
    else:
        warnings.warn("Target type cannot be determined")

    # Filter plot dict
    if l_plots is not None:
        d_calls = {x: d_calls[x] for x in l_plots}

    return d_calls


def plot_variable_importance(ax,
                             features, importance,
                             importance_cum=None, importance_mean=None, importance_error=None, max_score_diff=None,
                             category=None,
                             category_label="Importance",
                             category_color_palette=sns.xkcd_palette(["blue", "orange", "red"]),
                             color_error="grey"):
    """
    # Plot permutation based variable importance.

    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    feature: Numpy array or Pandas series
        Feature names.
    importance: Numpy array or Pandas series
        Importance of features, used for plotted bars.
    importance_cum: Numpy array or Pandas series
        Cumulated importance of features, used for overlayed line plot.
    importance_mean: Numpy array or Pandas series
        Mean importance of features, used for overlayed marker plot.        
    importance_error: Numpy array or Pandas series
        Error of mean importance of features, used for overlayed error lines added to markers.  
    max_score_diff: float
        Optional information which informs what an importance of 100 means in terms of score difference.
    category: Numpy array or Pandas series
        Grouping information used for coloring bars.
    category_label: str
        Used for title of legend.
    category_color_palette: seaborn color palette
        Colors of bars due to grouping by catgory.
    color_error: str
        Color of error bars.

    Returns
    -------
    Nothing
    """

    sns.barplot(x=importance, y=features, hue=category,
                palette=category_color_palette, dodge=False, ax=ax)
    ax.set_title(f"Top{len(features): .0f} Feature Importances")
    ax.set_xlabel(r"permutation importance")
    if max_score_diff is not None:
        ax.set_xlabel(ax.get_xlabel() + " (100 = " + str(max_score_diff) + r" score-$\Delta$)")
    if importance_cum is not None:
        ax.plot(importance_cum, features, color="black", marker="o")
        ax.set_xlabel(ax.get_xlabel() + " /\n" + r"cumulative in % (-$\bullet$-)")
    if importance_error is not None:
        ax.errorbar(x=importance_mean if importance_mean is not None else importance, 
                    y=features, 
                    xerr=importance_error,
                    linestyle="none", marker="s", fillstyle="none", color=color_error)
        ax.set_title(ax.get_title() + r" (incl. error (-$\boxminus$-))")

    '''
    if column_score_diff is not None:
        ax2 = ax.twiny()
        ax2.errorbar(x=df_varimp[column_score_diff], y=df_varimp[column_feature],
                     xerr=df_varimp[column_score_diff_error]*5,
                    fmt=".", marker="s", fillstyle="none", color="grey")
        ax2.grid(False)
    '''


def plot_pd(ax, feature_name, feature, yhat, feature_ref=None, yhat_err=None, refline=None, ylim=None,
            color="red", min_width=0.2):
    """
    # Plot partial dependence.

    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    feature_name: str
        Name of feature to use in axis.
    feature: Numpy array or Pandas series
        Feature values.
    yhat: Numpy array or Pandas series
        Average prediction, i.e. partial dependence value.
    feature_ref: Numpy array or Pandas series
        Reference feature values used for overlayed distribution plot.        
    yhat_err: Numpy array or Pandas series
        Error of yhat, used for error bands.  
    refline: float
        Reference line, usually base rate.
    ylim: tuple: (float, float)
        Limits y axis.
    color: str
        Color of partial dependence plot.
    min_width: float
        Minimum width of bars for categorical feature.

    Returns
    -------
    Nothing
    """

    print("plot PD for feature " + feature_name)
    numeric_feature = pd.api.types.is_numeric_dtype(feature)
    # if yhat.ndim == 1:
    #    yhat = yhat.reshape(-1, 1)

    if numeric_feature:

        # Lineplot
        ax.plot(feature, yhat, marker=".", color=color)

        # Background density plot
        if feature_ref is not None:
            ax2 = ax.twinx()
            ax2.axis("off")
            sns.kdeplot(feature_ref, color="grey",
                        shade=True, linewidth=0,  # hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 0},
                        ax=ax2)
        # Rugs
        sns.rugplot(feature, color="grey", ax=ax)

        # Refline
        if refline is not None:
            ax.axhline(refline, linestyle="dotted", color="black")  # priori line

        # Axis style
        ax.set_title(feature_name)
        ax.set_xlabel("")
        ax.set_ylabel(r"$\^y$")
        if ylim is not None:
            ax.set_ylim(ylim)

        # Crossvalidation
        if yhat_err is not None:
            ax.fill_between(feature, yhat - yhat_err, yhat + yhat_err, color=color, alpha=0.2)

    else:
        # Use DataFrame for calculation
        df_plot = pd.DataFrame({feature_name: feature, "yhat": yhat}).sort_values(feature_name).reset_index(drop=True)
        if yhat_err is not None:
            df_plot["yhat_err"] = yhat_err

        # Distribution
        if feature_ref is not None:
            df_plot = df_plot.merge(_helper_calc_barboxwidth(feature_ref, np.tile(1, len(feature_ref)),
                                                             min_width=min_width),
                                    how="inner")
            '''
            df_plot = df_plot.merge(pd.DataFrame({feature_name: feature_ref}).assign(count=1)
                                    .groupby(feature_name, as_index=False)[["count"]].sum()
                                    .assign(pct=lambda x: x["count"] / x["count"].sum())
                                    .assign(width=lambda x: 0.9 * x["pct"] / x["pct"].max()), how="left")
            df_plot[feature_name] = df_plot[feature_name] + " (" + (df_plot["pct"] * 100).round(1).astype(str) + "%)"
            if min_width is not None:
                df_plot["width"] = np.where(df_plot["width"] < min_width, min_width, df_plot["width"])
            #ax2 = ax.twiny()
            #ax2.barh(df_plot[feature_name], df_plot["pct"], color="grey", edgecolor="grey", alpha=0.5, linewidth=0)
            '''

        # Bar plot
        ax.barh(df_plot[feature_name] if feature_ref is None else df_plot[feature_name + "_fmt"],
                df_plot["yhat"],
                height=df_plot["w"] if feature_ref is not None else 0.8,
                color=color, edgecolor="black", alpha=0.5, linewidth=1)

        # Refline
        if refline is not None:
            ax.axvline(refline, linestyle="dotted", color="black")  # priori line

        # Inner barplot
        if feature_ref is not None:
            _helper_inner_barplot(ax, 
                                  x=df_plot[feature_name + "_fmt"],
                                  y=df_plot["pct"], inset_size=0.2)

        # Axis style
        ax.set_title(feature_name)
        ax.set_xlabel(r"$\^y$")
        if ylim is not None:
            ax.set_xlim(ylim)

        # Crossvalidation
        if yhat_err is not None:
            ax.errorbar(df_plot["yhat"],
                        df_plot[feature_name] if feature_ref is None else df_plot[feature_name + "_fmt"],
                        xerr=df_plot["yhat_err"],
                        linestyle="none", marker="s", capsize=5, fillstyle="none", color="grey")


# TODO: change color to tuple
def plot_shap(ax, shap_values, index, id,
              y_str=None, yhat_str=None,
              show_intercept=True, show_prediction=True,
              shap_lim=None,
              color=("blue", "red"), n_top=10, multiclass_index=None):
    """
    # Plot shapley values.

    Parameters
    ----------
    ax: matplotlib ax
        Ax which should be used by plot.
    shap_values: dict
        Shap values as returned from any SHAP explainer.
    index: int
        Index of shap_values to plot.
    id: str
        Label used in title.        
    y_str: str
        True label used in title.  
    yhat_str: str
        Predicted label used in title.  
    show_intercept: boolean
        Add bar for intercept.
    show_prediction: boolean
        Add bar for final prediction.
    shap_lim: tuple: (float, float)
        Limits of shap value axis.
    color: 2-tuple
        Color used for negative and postive shap_values respecitvely.
    n_top: int
        Number of shap_values to plot. Rest is aggregated to "... the rest" category.
    multiclass_index: int
        Which multiclass label should be printed.

    Returns
    -------
    Nothing
    """    

    # Subset in case of multiclass
    if multiclass_index is not None:
        base_values = shap_values.base_values[:, multiclass_index]
        values = shap_values.values[:, :, multiclass_index]
    else:
        base_values = shap_values.base_values
        values = shap_values.values

    # Shap values to dataframe
    df_shap = (pd.concat([pd.DataFrame({"variable": "intercept",
                                        "variable_value": np.nan,
                                        "shap": base_values[index]}, index=[0]),
                          pd.DataFrame({"variable": shap_values.feature_names[index],
                                        "variable_value": shap_values.data[index],
                                        "shap": values[index]})
                          .assign(tmp=lambda x: x["shap"].abs())
                          .sort_values("tmp", ascending=False)
                          .drop(columns="tmp")])
               .assign(yhat=lambda x: x["shap"].cumsum()))  # here a my.inv_logit might be added

    # Prepare for waterfall plot
    df_plot = (df_shap.assign(offset=lambda x: x["yhat"].shift(1).fillna(0),
                              bar=lambda x: x["yhat"] - x["offset"],
                              color=lambda x: np.where(x["variable"] == "intercept", "grey",
                                                       np.where(x["bar"] > 0, color[1], color[0])),
                              bar_label=lambda x: np.where(x["variable"] == "intercept",
                                                           x["variable"],
                                                           x["variable"] + " = " + x["variable_value"].astype("str")))
               .loc[:, ["bar_label", "bar", "offset", "color"]])

    # Aggreagte non-n_top shap values
    if n_top is not None:
        df_plot = pd.concat([df_plot.iloc[:(n_top + 1)],
                            pd.DataFrame(dict(bar_label="... the rest",
                                              bar=df_plot.iloc[(n_top + 1):]["bar"].sum(),
                                              offset=df_plot.iloc[(n_top + 1)]["offset"]),
                                         index=[0])
                            .assign(color=lambda x: np.where(x["bar"] > 0, color[1], color[0]))])

    # Add final prediction
    df_plot = (pd.concat([df_plot, pd.DataFrame(dict(bar_label="prediction", bar=df_plot["bar"].sum(),
                                                     offset=0, color="black"), index=[0])])
               .reset_index(drop=True))

    # Remove intercept and final prediction
    if not show_intercept:
        df_plot = df_plot.query("bar_label != 'intercept'")
    if not show_prediction:
        df_plot = df_plot.query("bar_label != 'prediction'")
    #df_plot = df_plot.query('bar_label not in ["intercept", "Prediction"]')
    #x_min = (df_plot["offset"]).min()
    #x_max = (df_plot["offset"]).max()
    x_min = min(0, (df_plot["offset"]).min())
    x_max = max(0, (df_plot["offset"]).max())

    # Plot bars
    ax.barh(df_plot["bar_label"], df_plot["bar"], left=df_plot["offset"], color=df_plot["color"],
            alpha=0.5,
            edgecolor="black")

    # Set axis limits
    if shap_lim is not None:
        x_min = shap_lim[0]
        x_max = shap_lim[1]
    ax.set_xlim(x_min - 0.1 * (x_max - x_min),
                x_max + 0.1 * (x_max - x_min))

    # Annotate
    for i in range(len(df_plot)):
        # Text
        ax.annotate(df_plot.iloc[i]["bar"].round(3),
                    (df_plot.iloc[i]["offset"] + max(0, df_plot.iloc[i]["bar"]) + np.ptp(ax.get_xlim()) * 0.02,
                     df_plot.iloc[i]["bar_label"]),
                    # if ~df_plot.iloc[i][["bar_label"]].isin(["intercept", "Prediction"])[0] else "right",
                    ha="left",
                    va="center", size=10,
                    color="black")  # "white" if i == (len(df_plot) - 1) else "black")

        # Lines
        if i < (len(df_plot) - 1):
            df_line = pd.concat([pd.DataFrame(dict(x=df_plot.iloc[i]["offset"] + df_plot.iloc[i]["bar"],
                                                   y=df_plot.iloc[i]["bar_label"]), index=[0]),
                                pd.DataFrame(dict(x=df_plot.iloc[i]["offset"] + df_plot.iloc[i]["bar"],
                                                  y=df_plot.iloc[i + 1]["bar_label"]), index=[0])])
            ax.plot(df_line["x"], df_line["y"], color="black", linestyle=":")

    # Title and labels
    title = "id = " + str(id)
    if y_str is not None:
        title = title + " (y = " + y_str + ")"
    if yhat_str is not None:
        title = title + r" ($\^ y$ = " + yhat_str + ")"
    ax.set_title(title)
    ax.set_xlabel("shap")
