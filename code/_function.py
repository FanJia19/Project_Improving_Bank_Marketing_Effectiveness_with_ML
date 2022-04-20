import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt


def import_dataset(filename):
    """
    Import the dataset from the path.

    Parameters
    ----------
        filename : str
            filename with path

    Returns
    -------
        data : DataFrame

    Examples
    --------
        bank_mkt = import_dataset("../data/BankMarketing.csv")
    """
    bank_mkt = pd.read_csv(
        filename,
        # Additional strings to recognize as NA/NaN. 
        na_values=["unknown", "nonexistent"],
        # Values to consider as True, list
        true_values=["yes", "success"],
        # Values to consider as False, list
        false_values=["no", "failure"],
    )
    # Treat pdays = 999 as missing values
    bank_mkt["pdays"] = bank_mkt["pdays"].replace(999, pd.NA)
    
    # `month` will be encoded to the corresponding number, e.g. "mar" -> 3
    month_map = {
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    
    # Replace month strings with numbers
    bank_mkt["month"] = bank_mkt["month"].replace(month_map)
    
    # `day_of_week` will be encoded to the corresponding number, e.g. "wed" -> 3
    dow_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}
    bank_mkt["day_of_week"] = bank_mkt["day_of_week"].replace(dow_map)
    
    # Convert types, "Int64" is nullable integer data type in pandas
    bank_mkt = bank_mkt.astype(
        dtype={
            "age": "Int64",
            "job": "category",
            "marital": "category",
            "education": "category",
            "default": "boolean",
            "housing": "boolean",
            "loan": "boolean",
            "contact": "category",
            "month": "Int64",
            "day_of_week": "Int64",            
            "duration": "Int64",
            "campaign": "Int64",
            "pdays": "Int64",
            "previous": "Int64",
            "poutcome": "boolean",
            "y": "boolean",
        }
    )
    
    # Drop 12 duplicated rows
    bank_mkt = bank_mkt.drop_duplicates().reset_index(drop=True)
    
    # reorder ordinal categorical data
    bank_mkt["education"] = bank_mkt["education"].cat.reorder_categories(
        [
            "illiterate",
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "professional.course",
            "university.degree",
        ],
        ordered=True,
    )
    return bank_mkt


def split_dataset(data, preprocessor=None, random_state=62):
    """
    Split dataset into train, test and validation sets using preprocessor.
    Because the random state of validation set is not specified, the validation set will be different each time when the function is called.

    Parameters
    ----------
        data : DataFrame

        preprocessor : Pipeline

        random_state : int

    Returns
    -------
        datasets : tuple

    Examples
    --------
        from sklearn.preprocessing import OrdinalEncoder
        data = import_dataset("../data/BankMarketing.csv").interpolate(method="pad").loc[:, ["job", "education", "y"]]
        # To unpack all train, test, and validation sets
        X_train, y_train, X_test, y_test, X_ttrain, y_ttrain, X_validate, y_validate = split_dataset(data, OrdinalEncoder())
        # To unpack train and test sets.
        X_train, y_train, X_test, y_test, *other_sets = split_dataset(data, OrdinalEncoder())
        # To unpack test and validation set
        *other_sets, X_test, y_test, X_ttrain, y_ttrain, X_validate, y_validate = split_dataset(data, OrdinalEncoder())
        # To unpack only train set.
        X_train, y_train, *other_sets = split_dataset(data, OneHotEncoder())
    """
    train_test_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=random_state
    )
    for train_index, test_index in train_test_split.split(
        data.drop("y", axis=1), data["y"]
    ):
        train_set = data.iloc[train_index]
        test_set = data.iloc[test_index]

    X_train = train_set.drop(["duration", "y"], axis=1)
    y_train = train_set["y"].astype("int").to_numpy()
    X_test = test_set.drop(["duration", "y"], axis=1)
    y_test = test_set["y"].astype("int").to_numpy()

    train_validate_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for ttrain_index, validate_index in train_validate_split.split(X_train, y_train):
        ttrain_set = train_set.iloc[ttrain_index]
        validate_set = train_set.iloc[validate_index]

    X_ttrain = ttrain_set.drop(["duration", "y"], axis=1)
    y_ttrain = ttrain_set["y"].astype("int").to_numpy()
    X_validate = validate_set.drop(["duration", "y"], axis=1)
    y_validate = validate_set["y"].astype("int").to_numpy()

    if preprocessor != None:
        X_train = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        X_ttrain = preprocessor.fit_transform(X_ttrain, y_ttrain)
        X_validate = preprocessor.transform(X_validate)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_ttrain,
        y_ttrain,
        X_validate,
        y_validate,
    )


def benchmark(data, preprocessor=None, clf=None):
    """
    Benchmark preprocessor and clf's performance on train, validation and test sets.
    All the data transformation should be handled by preprocessor and estimation should be handled by clf.

    Parameters
    ----------
        data : DataFrame

        preprocessor : Pipeline, default = None

        clf : estimator, default = None

    """
    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_ttrain,
        y_ttrain,
        X_validate,
        y_validate,
    ) = split_dataset(data, preprocessor)
    X_sets = [X_ttrain, X_validate, X_test]
    y_sets = [y_ttrain, y_validate, y_test]

    metric_names = ["TNR", "TPR", "bACC", "ROC", "REC", "PRE", "AP"]
    set_names = ["Train", "Validate", "Test"]
    metric_df = pd.DataFrame(index=metric_names, columns=set_names)

    try:
        clf.fit(X_ttrain, y_ttrain, eval_set=(X_validate, y_validate), verbose=False)
    except (ValueError, TypeError, KeyError):
        clf.fit(X_ttrain, y_ttrain)

    for name, X, y in zip(set_names, X_sets, y_sets):
        # Re-fit model on train set before test set evaluation except CatBoost
        if name == "Test" and not isinstance(clf, CatBoostClassifier):
            clf.fit(X_train, y_train)
        y_pred = clf.predict(X)

        try:
            y_score = clf.decision_function(X)
        except AttributeError:
            y_score = clf.predict_proba(X)[:, 1]

        metrics = [
            recall_score(y, y_pred, pos_label=0),
            recall_score(y, y_pred),
            balanced_accuracy_score(y, y_pred),
            roc_auc_score(y, y_score),
            recall_score(y, y_pred),
            precision_score(y, y_pred),
            average_precision_score(y, y_score),
        ]
        metric_df[name] = metrics

    return metric_df


def render_benchmark(
    data,
    col_width=3.0,
    row_height=0.625,
    font_size=14,
    header_color="#334074",
    row_colors=["#f1f1f2", "w"],
    edge_color="w",
    bbox=[0, 0, 1, 1],
    header_columns=0,
    ax=None,
    **kwargs
):
    """
    Reference
    ---------
    # https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure

    Examples
    --------
    logit_best = benchmark(bank_mkt, hot_transformer, lrmodel_gs2)
    fig,ax = render_benchmark(logit_best, header_columns=0, col_width=1.5)
    fig.savefig("logit_best.png")
    """
    data = data.round(decimals=3)
    # data = data.rename({"index": "Metrics"}, axis=1)
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array(
            [col_width, row_height]
        )
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")
    mpl_table = ax.table(
        cellText=data.values,
        bbox=bbox,
        colLabels=data.columns,
        rowLabels=data.index,
        colLoc="left",
        rowLoc="right",
        **kwargs
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor(header_color)
        elif k[1] < header_columns:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax.get_figure(), ax


def dftransform(
    X,
    drop=None,
    cut=None,
    gen=None,
    cyclic=None,
    target=None,
    fillna=True,
    to_float=False,
):
    """
    Encode, transform, and generate categorical data in the dataframe.

    Parameters
    ----------
        X : DataFrame

        drop : list, default = None

        gen : list, default = None

        cut : list, default = None

        external : list, default = None

        cyclic : list, default = None

        fillna : boolean, default = True

        to_float : boolean, default = False

    Returns
    -------
        X : DataFrame

    Examples
    --------
    bank_mkt = import_dataset("../data/BankMarketing.csv")
    X = dftransform(bank_mkt)
    """
    X = X.copy()

    if gen != None:
        if "year" in gen or "days" in gen:
            X.loc[X.index < 27682, "year"] = 2008
            X.loc[(27682 <= X.index) & (X.index < 39118), "year"] = 2009
            X.loc[39118 <= X.index, "year"] = 2010
            X["year"] = X["year"].astype("int")
        if "days" in gen:
            X["date"] = pd.to_datetime(X[["month", "year"]].assign(day=1))
            X["lehman"] = pd.to_datetime("2008-09-15")
            X["days"] = X["date"] - X["lehman"]
            X["days"] = X["days"].dt.days
            X = X.drop(["lehman", "year", "date"], axis=1)
        if "has_previous" in gen:
            X["has_previous"] = X["previous"] > 0
        if "has_default" in gen:
            X["has_default"] = X["default"].notna()
        if "has_marital" in gen:
            X["has_marital"] = X["marital"].notna()

    if cut != None:
        if "pdays" in cut:
            # Cut pdays into categories
            X["pdays"] = pd.cut(
                X["pdays"],
                [0, 3, 5, 10, 15, 30, 1000],
                labels=[3, 5, 10, 15, 30, 1000],
                include_lowest=True,
            ).astype("Int64")

    if cyclic != None:
        if "month" in cyclic:
            X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
            X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
            X = X.drop("month", axis=1)
        if "day_of_week" in cyclic:
            X["day_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 5)
            X["day_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 5)
            X = X.drop("day_of_week", axis=1)

    # Transform target encoded feature as str
    if target != None:
        X[target] = X[target].astype("str")

    # Other categorical features will be coded as its order in pandas categorical index
    X = X.apply(
        lambda x: x.cat.codes
        if pd.api.types.is_categorical_dtype(x)
        else (x.astype("Int64") if pd.api.types.is_bool_dtype(x) else x)
    )

    if fillna:
        # Clients who have been contacted but do not have pdays record should be encoded as 999
        # Clients who have not been contacted should be encoded as -999
        X.loc[X["pdays"].isna() & X["poutcome"].notna(), "pdays"] = 999
        X["pdays"] = X["pdays"].fillna(-999)
        # Fill other missing values as -1
        X = X.fillna(-1)
    else:
        X = X.astype("float")

    if drop != None:
        # Drop features
        X = X.drop(drop, axis=1)

    if to_float:
        X = X.astype("float")

    return X
