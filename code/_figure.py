from _function import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer

# Cosmetic options
rc = {
    "figure.figsize": (6.4, 4.8),
    "figure.dpi": 300,
    "axes.titlesize": "large",
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.titlelocation": "left",
}

sns.set_theme(context="notebook", style="darkgrid", color_codes=True, rc=rc)

# Exploratory Data Analysis
def eda_figures():
    # 2_1_Visualise Y distribution
    bank_mkt = import_dataset("data/BankMarketing.csv")
    y_count = bank_mkt["y"].value_counts().plot(kind="bar")
    plt.savefig("docs/figures/2_1_Y_distribution.png")

    # 2_2_Uneven Distribution of Positive Outcome
    bank_mkt = import_dataset("data/BankMarketing.csv")
    bank_mkt["year"] = 2008
    bank_mkt.loc[27682:, "year"] = 2009
    bank_mkt.loc[39118:, "year"] = 2010
    bank_mkt["date"] = pd.to_datetime(bank_mkt[["month", "year"]].assign(day=1))
    p = bank_mkt[bank_mkt.y == True].reset_index()
    p.loc[(p.month == 10) & (p.year == 2008), "year"] = "Financial Crisis"
    ax = sns.histplot(
        data=p,
        x="index",
        stat="count",
        hue="year",
        bins=500,
        palette="deep",
        legend=True,
    )
    ax.get_legend().set_title("")
    ax.set_ylim(0, 60)
    fig = ax.set(xlabel="", ylabel="")
    fig = ax.get_figure()
    fig.savefig("docs/figures/2_2_Uneven_distribution.png")

    # 2_3_Postive rate by month
    bank_mkt[["date", "y"]].groupby("date").mean().plot.line(ylabel="", legend=False)
    plt.savefig("docs/figures/2_3_Positive_rate_by_month.png")

    # 2_4_Five economic indicators
    econ_df = bank_mkt.iloc[:, 15:]
    econ_df = econ_df.drop("y", axis=1)
    sc_econ = (econ_df.iloc[:, 0:5] - econ_df.iloc[:, 0:5].min()) / (
        econ_df.iloc[:, 0:5].max() - econ_df.iloc[:, 0:5].min()
    )
    sc_econ_df = pd.concat([sc_econ, econ_df.iloc[:, -1:]], axis=1)
    plt.rcParams["figure.figsize"] = (10, 5)
    fig, ax = plt.subplots()
    ax.plot(sc_econ_df["date"], sc_econ_df["emp.var.rate"])
    ax.plot(sc_econ_df["date"], sc_econ_df["cons.price.idx"])
    ax.plot(sc_econ_df["date"], sc_econ_df["cons.conf.idx"])
    ax.plot(sc_econ_df["date"], sc_econ_df["euribor3m"])
    ax.plot(sc_econ_df["date"], sc_econ_df["nr.employed"])
    ax.legend(
        ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
        bbox_to_anchor=(1, 1),
    )
    ax.xaxis.set_label_text("Date")
    ax.yaxis.set_label_text("Value")
    plt.savefig("docs/figures/2_4_Five_econ_indicators.png")

    # 2_5_Missing_value_percentage
    na = bank_mkt.isna().sum()
    na_nonzero = na[na != 0]
    na_perc = na_nonzero / bank_mkt.y.count()
    na_bar = na_perc.plot.bar()
    plt.savefig("docs/figures/2_5_Missing_value_percentage.png")

    # 2_6_Age_distribution
    age_hist = bank_mkt["age"].plot.hist()
    age_box = bank_mkt["age"].plot.box(vert=False, sym=".")
    plt.savefig("docs/figures/2_6_Age_box.png")

    # 2_7_Job
    job_outcome = cat_outcome(bank_mkt, "job")
    plt.savefig("docs/figures/2_7_Job.png")

    # 2_8_Education
    marital_outcome = cat_outcome(bank_mkt, "marital")
    plt.savefig("docs/figures/2_8_Education.png")

    # 2_9_Default
    default_outcome = cat_outcome(bank_mkt, "default")
    plt.savefig("docs/figures/2_9_Default.png")

    # 2_10_Contact
    contact_outcome = cat_outcome(bank_mkt, "contact")
    plt.savefig("docs/figures/2_10_Contact.png")

    # 2_11_Month
    month_outcome = cat_outcome(bank_mkt, "month")
    plt.savefig("docs/figures/2_11_Month.png")

    # 2_12_Pdays
    day_outcome = cat_outcome(bank_mkt, "pdays")
    plt.savefig("docs/figures/2_12_Pdays.png")

    # 2_13_Previous
    day_outcome = cat_outcome(bank_mkt, "previous")
    plt.savefig("docs/figures/2_13_Previous.png")

    # 2_14_Poutcome + 2_14_Pdays+Previous
    bank_mkt2["poutcome"].value_counts().plot(kind="bar")
    plt.savefig("docs/figures/2_14_Poutcome.png", bbox_inches="tight")
    previous_na = bank_mkt[["pdays", "poutcome"]].isna().sum()
    previous_na_ax = previous_na.plot.bar()
    plt.savefig("docs/figures/2_14_Pdays+Previous.png", bbox_inches="tight")

    # 2_15_Job+Marital
    job_marital_total = (
        bank_mkt[["job", "marital", "y"]]
        .groupby(["job", "marital"])
        .count()
        .y.unstack()
    )
    job_marital_true = (
        bank_mkt[["job", "marital", "y"]].groupby(["job", "marital"]).sum().y.unstack()
    )
    job_marital_rate = job_marital_true / job_marital_total
    job_marital_rate = job_marital_rate.rename_axis(None, axis=0).rename_axis(
        None, axis=1
    )
    job_marital_heatmap = sns.heatmap(
        data=job_marital_rate, vmin=0, vmax=0.5, annot=True
    )
    plt.savefig("docs/figures/2_15_Job+Marital.png", bbox_inches="tight")

    # 2_15_Job+Education
    job_education_total = (
        bank_mkt[["job", "education", "y"]]
        .groupby(["job", "education"])
        .count()
        .y.unstack()
    )
    job_education_true = (
        bank_mkt[["job", "education", "y"]]
        .groupby(["job", "education"])
        .sum()
        .y.unstack()
    )
    job_education_rate = job_education_true / job_education_total
    job_education_rate = job_education_rate.rename_axis(None, axis=0).rename_axis(
        None, axis=1
    )
    job_education_heatmap = sns.heatmap(
        data=job_education_rate, vmin=0, vmax=0.5, annot=True
    )
    plt.savefig("docs/figures/2_15_Job+Education.png", bbox_inches="tight")

    # 2_15_Education+Marital
    education_marital_total = (
        bank_mkt[["education", "marital", "y"]]
        .groupby(["education", "marital"])
        .count()
        .y.unstack()
    )
    education_marital_true = (
        bank_mkt[["education", "marital", "y"]]
        .groupby(["education", "marital"])
        .sum()
        .y.unstack()
    )
    education_marital_rate = education_marital_true / education_marital_total
    education_marital_rate = education_marital_rate.rename_axis(
        None, axis=0
    ).rename_axis(None, axis=1)
    education_marital_heatmap = sns.heatmap(
        data=education_marital_rate, vmin=0, vmax=0.5, annot=True
    )
    plt.savefig("docs/figures/2_15_Education+Marital.png", bbox_inches="tight")

    # 2_17_Heatmap
    # corr_heatmap = sns.heatmap(data=bank_mkt.corr(method="pearson"))
    # plt.savefig("docs/figures/2_17_Heatmap.png", bbox_inches="tight")

    bank_mkt = import_dataset("data/BankMarketing.csv")
    corr_heatmap = sns.heatmap(data=bank_mkt.corr(method="pearson"))
    plt.savefig("docs/figures/2_17_Heatmap.png", bbox_inches="tight")


# Feature Engineering
def conf_mat_annot():
    f, ax = plt.subplots(figsize=(4.8, 4.8))
    conf_mat = np.array([[100, 30], [30, 100]])
    conf_label = np.array([["TN", "FP"], ["FN", "TP"]])
    conf_ax = sns.heatmap(
        conf_mat,
        ax=ax,
        annot=conf_label,
        annot_kws={"fontweight": "bold"},
        fmt="",
        square=True,
        cmap=plt.cm.Blues,
        cbar=False,
    )
    conf_ax.set_xlabel("Predicted")
    conf_ax.set_ylabel("True")
    f.savefig("../docs/figures/5_1_Conf_Mat.png", bbox_inches="tight")


def conf_mat_knn():
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix

    bank_mkt = import_dataset("../data/BankMarketing.csv")
    preprocessor = FunctionTransformer(dftransform)
    X_train, y_train, *other_sets = split_dataset(bank_mkt, preprocessor)
    clf = KNeighborsClassifier(n_neighbors=10)
    y_pred = cross_val_predict(clf, X_train, y_train, cv=5)
    conf_mat = confusion_matrix(y_train, y_pred)
    f, ax = plt.subplots(figsize=(4.8, 4.8))
    conf_ax = sns.heatmap(
        conf_mat, ax=ax, annot=True, fmt="", cmap=plt.cm.Blues, cbar=False
    )
    conf_ax.set_xlabel("Predicted")
    conf_ax.set_ylabel("True")
    f.savefig("../docs/figures/5_2_Conf_Mat_KNN.png", bbox_inches="tight")


def conf_plot(clf):
    f, ax = plt.subplots(figsize=(4.8, 4.8))
    y_test_pred = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_test_pred)
    conf_ax = sns.heatmap(
        conf_mat, ax=ax, annot=True, fmt="", cmap=plt.cm.Blues, cbar=False
    )
    conf_ax.set_xlabel("Predicted")
    conf_ax.set_ylabel("True")
    f.savefig("../docs/figures/Conf_Mat.png", bbox_inches="tight")


def pre_rec_threshold():
    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve

    bank_mkt = import_dataset("../data/BankMarketing.csv")
    preprocessor = FunctionTransformer(dftransform)
    X_train, y_train, *other_sets = split_dataset(bank_mkt, preprocessor)
    clf = LogisticRegression(class_weight="balanced")
    y_score = cross_val_predict(clf, X_train, y_train, cv=5, method="decision_function")
    f, ax = plt.subplots()
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_score)
    pre_rec_df = pd.DataFrame(
        {"Precision": precisions[:-1], "Recall": recalls[:-1]}, index=thresholds
    )
    pre_rec_ax = pre_rec_df.plot.line(ax=ax, ylim=(-0.05, 1.05))
    threshold = 0
    pre_rec_ax.plot((threshold, threshold), (-2, 2), linestyle="--", linewidth=1)
    f.savefig("../docs/figures/5_3_Pre_Rec_Logi.png", bbox_inches="tight")


def tree_importance():
    columns = X_train.columns.tolist()
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [columns[i] for i in indices])
    plt.show()


if __name__ == "__main__":
    # eda_figures()
    conf_mat_annot()
    conf_mat_knn()
    pre_rec_threshold()
