# Feature Engineering
 ```{=latex}
\chapterauthor{Jiawei Li}
```

Feature Engineering is a practice of using domain knowledge to incorporate more features and improve machine learning models. Given that the data set has relatively few features and rows, dimensionality reduction will not be applied. We instead focus on improving feature quality and generating new features, such as incorporating dates. As a result, our feature engineering approach includes missing value imputation, feature generation, feature transformations and adjusting sample weights.

## Incorporate Dates

The most important feature in the data set is the date of telemarketing which gives us the economic context. However, the data set only includes `month` and `day_of_week` while `year` is missing. With the help of the data set description, we find that the data set is ordered by date and the telemarketing was conducted from May 2008 to November 2010. Even without this description, we can still reverse engineer the year information using the economic index, such as `CPI`. By manually inspecting `month`, we can infer each row's `year`.

```python
bank_mkt = import_data set("../data/BankMarketing.csv")
bank_mkt.loc[bank_mkt.index < 27682, "year"] = 2008
bank_mkt.loc[(27682<=bank_mkt.index) & (bank_mkt.index<39118), "year"] = 2009
bank_mkt.loc[39118<=bank_mkt.index, "year"] = 2010
bank_mkt["year"] = bank_mkt["year"].astype("int")
```

With `year` and `month`, we can approximate each marketing call's date as the start of the month.

```python
bank_mkt["date"] = pd.to_datetime(bank_mkt[["month", "year"]].assign(day=1))
```

Simply feeding `date` into the models neither incorporates much information nor improve the model performance. Setting focal points instead is a much better way to treat date information. As discussed in chapter 2, there is a surge in success rate after the financial crisis. Therefore, we use the date when Lehman Brothers filed bankruptcy as the focal point and create a new feature `days` which is the days before or after Lehman Brothers bankruptcy.

```python
bank_mkt["lehman"] = pd.to_datetime("2008-09-15")
bank_mkt["days"] = bank_mkt["date"] -  bank_mkt["lehman"]
bank_mkt["days"] = bank_mkt["days"].dt.days
```

## Impute Missing Values

There are several strategies to handle missing values. The simplest way is to impute missing value as a different category, such as `-1`, depends on the context. For categorical data, `-1` is used. For `pdays`, both `999` and `-999` are used. Clients who have been contacted but do not have `pdays` record should be encoded as `999`, while clients who have not been contacted should be encoded as `-999`.

```python
# Clients who have been contacted but do not have pdays record should be encoded as 999
bank_mkt.loc[bank_mkt["pdays"].isna() & bank_mkt["poutcome"].notna(), "pdays"] = 999
# Clients who have not been contacted should be encoded as -999 
bank_mkt["pdays"] = bank_mkt["pdays"].fillna(-999)
# Fill other missing values as -1
bank_mkt = bank_mkt.fillna(-1)
```

It is also possible to impute missing values as the most frequent value using `SimpleImputer`.

```python
from sklearn.impute import SimpleImputer

freq_features = ["job", "marital", "education", "default", "housing", "loan"]

freq_imputer = ColumnTransformer([
    ("freq_imputer", SimpleImputer(missing_values=-1, strategy="most_frequent"),
    freq_features)
], remainder="passthrough")

freq_encoder = make_pipeline(cat_encoder, freq_imputer)
X_train = freq_encoder.fit_transform(X_train, y_train)
X_test = freq_encoder.transform(X_test), axis=1)
```

Another imputation method worth mentioning is iterative imputation which attempts to estimate missing values. However, this approach may bring overfitting to the models.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

ite_features = ["age", 
                "job", 
                "marital", 
                "education", 
                "default", 
                "housing", 
                "loan", 
                "contact", 
                "campaign", 
                "month", 
                "day_of_week", 
                "pdays", 
                "previous"]

ite_imputer = ColumnTransformer([
    ("ite_imputer",
     make_pipeline(
         IterativeImputer(max_iter=100,
                          missing_values=-1,
                          initial_strategy="most_frequent",
                          random_state=42),
         FunctionTransformer(np.round)
     ),
     ite_features),
], remainder="passthrough")

ite_encoder = make_pipeline(cat_encoder, ite_imputer)
X_train = ite_encoder.fit_transform(X_train, y_train)
X_test = ite_encoder.transform(X_test), axis=1
```

## Drop Demographic Features
To our surprise, the biggest improvement is achieved by dropping demographic features. Several reasons contribute to this result. First, the financial crisis and the debt crisis in Portugal may have lowered people’s expectation of their future income and altered their investment choice. Thus, people reacted differently to the term deposit after the crisis and demographic data does not matter anymore. Second, we do not have enough data in 2009 and 2010 to learn this abnormal shift of attitude and therefore demographic data becomes noise.

```python
drop_features = ["age",
                 "job",
                 "marital",
                 "education",
                 "housing",
                 "loan",
                 "default",
                 "duration",
                 "y"]
bank_mkt = bank_mkt.drop(drop_features, axis=1)
```

## Feature Engineering in Practice
During our project, we implement different feature engineering strategies, such as target encoding and cyclic encoding, inside a function `dttransform()`. Each strategy can be called by passing its corresponding parameters and put into a preprocessing pipeline as demonstrated in the following code.

```python
drop_features = ["age",
                 "job",
                 "marital",
                 "education",
                 "housing",
                 "loan",
                 "default",
                 "duration",
                 "y"]
drop_prep = FunctionTransformer(dftransform, 
                                kw_args={"drop": drop_features})
```

Following this method, we build classifiers using different feature engineering strategies and compare them with their baseline performance. We then select the best feature engineering strategies and tune the hyperparameters for each classifier. However, it should be noted that feature engineering strategies can also be regarded as hyperparameters for the model. It is also possible to use the grid search to find the best feature engineering strategies and hyperparameters.
