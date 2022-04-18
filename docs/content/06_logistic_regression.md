# Logistic Regression
 ```{=latex}
\chapterauthor{Fan Jia}
```

Logistic Regression is commonly used to estimate the probability of an instance belonging to a particular class. If the probability is greater than 50%, the model will classify the instance to that class, otherwise, it will not. Therefore, Logistic Regression is a binary classifier that can be applied to our data set. Underlying the model is the logistic sigmoid function as shown below. This classifier can potentially perform very well on linearly separable classes.

$$l(x)= \frac{1}{1+e^{ -\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}+\theta_{3}x_{3}+...+\theta_{n}x_{n}}}$$

## Initialization 
First, we preprocess the data with one-hot encoding and standardization. Then the Logistic Regression need two important parameters:

1. `class weight = "balanced"`, which is necessary to handle our imbalanced data set; 
2. The maximum number of iterations taken for the solvers to converge.

```python
cat_features = ["job",
                "marital",
                "education",
                "default",
                "housing",
                "loan",
                "poutcome"]

num_features =  ["age",
                 "campaign",
                 "pdays",
                 "previous",
                 "emp.var.rate",
                 "cons.price.idx",
                 "cons.conf.idx",
                 "euribor3m",
                 "nr.employed"]

hot_scaler = ColumnTransformer([
    ("one_hot_encoder", OneHotEncoder(drop="first"), cat_features),
    ("scaler", StandardScaler(), num_features)
], remainder="passthrough")

hot_transformer = make_pipeline(FunctionTransformer(dftransform), hot_scaler)

X_train, y_train, X_test, y_test, *other_sets = split_dataset(bank_mkt, hot_transformer)

lrmodel = LogisticRegression(class_weight='balanced', max_iter=10000) 
```

## Grid Search
Next, we use grid search to find the optimal parameters for the model. For the first grid search, we pick two parameters: the penalty L2 and its inverse parameter C. The L1, L2 regularization parameters are used to avoid overfitting of data due to either collinearity or high-dimensionality. They both shrink the estimates of the regression coefficients towards zero. When two predictors are highly correlated, L1 will pick one of the two predictors, and in contrast, L2 will keep both of them and jointly shrink the coefficients together a little bit. Parameter C is the inverse of the regularization strength, with smaller values leading to stronger regularization. 

```python
# Try the 1st grid search param_grid combination:
lrmodel = LogisticRegression(class_weight='balanced', max_iter=10000)

# grid search
param_grid = {'penalty': ['l2'],
              'C':[0.001,.009,0.01,0.05,0.09,5,10,25,50,100]}
GS_lrmodel_1 = GridSearchCV(lrmodel, param_grid, scoring='average_precision', n_jobs=-1)
GS_lrmodel_1.fit(X_train, y_train)
lrmodel_gs1 = lrmodel.set_params(**GS_lrmodel_1.best_params_)

# Use calibrated model on train set
lrmodel_gs1.fit(X_train, y_train)
y_train_pred = lrmodel_gs1.predict(X_train)
y_train_score = lrmodel_gs1.decision_function(X_train)
cmtr_gs1 = confusion_matrix(y_train, y_train_pred)
acctr_gs1 = accuracy_score(y_train, y_train_pred)
aps_train_gs1 = average_precision_score(y_train, y_train_score)

# Test the model
lrmodel_gs1.fit(X_test, y_test)
y_test_pred = lrmodel_gs1.predict(X_test)
y_test_score = lrmodel_gs1.decision_function(X_test)
cmte_gs1 = confusion_matrix(y_test, y_test_pred)
accte_gs1 = accuracy_score(y_test, y_test_pred)
aps_test_gs1 = average_precision_score(y_test, y_test_score)
print('Confusion Matrix:\n',cmtr_gs1,'\nAccuracy Score:\n',acctr_gs1, '\nAPS:\n',aps_train_gs1)
print('Confusion Matrix:\n',cmte_gs1,'\nAccuracy Score:\n',accte_gs1, '\nAPS:\n',aps_test_gs1)
print('best parameters:',GS_lrmodel_1.best_params_)
```

The results show a slight improvement compared to the initial model, with a 0.7869 accuracy score, a 0.4418 average precision score and an ROC value of 0.783 for the test set. Additionally, this grid search finds `{'C': 10, 'penalty': 'l2'}` as the best parameter combination. The confusion matrices and performance measures are presented below.

|      | Train    | Validate | Test     |
| ---- | -------- | -------- | -------- |
| TNR  | 0.800539 | 0.808758 | 0.803229 |
| TPR  | 0.663523 | 0.669811 | 0.658405 |
| bACC | 0.732031 | 0.739285 | 0.730817 |
| ROC  | 0.785491 | 0.786075 | 0.782550 |
| REC  | 0.663523 | 0.669811 | 0.658405 |
| PRE  | 0.296955 | 0.307740 | 0.298194 |
| AP   | 0.434798 | 0.459581 | 0.441740 |
: Performance metrics of Logistic Regression with L2 regularization

![Confusion Matrix of Logistic Regression with L2 regularization](../figures/6_1_Conf_Mat_1.png){width=45%}

For the second grid search, we used the L1 penalty and Elasticnet penalty, which combines L1 and L2 penalties and will give a result in between. We also used the solver “saga”, which supports the non-smooth penalty L1 and is often used to handle the potential multinomial loss in the regression.

```python
# Try the 2nd grid search param_grid combination
lrmodel_gs = LogisticRegression(class_weight='balanced',max_iter=10000)

# grid search
param_grid = {"C":[0.001,.009,0.01,0.05,0.09,5,10,25,50,100], 
              "penalty":["l1","elasticnet"],
              "solver": ["saga"]}
GS_lrmodel_2 = GridSearchCV(lrmodel_gs, param_grid, scoring='average_precision', n_jobs=-1)
GS_lrmodel_2.fit(X_train, y_train)
lrmodel_gs2 = lrmodel_gs.set_params(**GS_lrmodel_2.best_params_)

# Use calibrated model on train set
lrmodel_gs2.fit(X_train, y_train)
y_train_pred = lrmodel_gs2.predict(X_train)
y_train_score = lrmodel_gs1.decision_function(X_train)
cmtr_gs2 = confusion_matrix(y_train, y_train_pred)
acctr_gs2 = accuracy_score(y_train, y_train_pred)
aps_train_gs2 = average_precision_score(y_train, y_train_pred)

# Test the model
lrmodel_gs2.fit(X_test, y_test)
y_test_pred = lrmodel_gs2.predict(X_test)
y_test_score = lrmodel_gs1.decision_function(X_test)
cmte_gs2 = confusion_matrix(y_test, y_test_pred)
accte_gs2 = accuracy_score(y_test, y_test_pred)
aps_test_gs2 = average_precision_score(y_test, y_test_score)
print('Confusion Matrix:\n',cmtr_gs2,'\nAccuracy Score:\n',acctr_gs1, '\nAPS:\n',aps_train_gs1)
print('Confusion Matrix:\n',cmte_gs2,'\nAccuracy Score:\n',accte_gs2, '\nAPS:\n',aps_test_gs2)
print('best parameters:',GS_lrmodel_2.best_params_)
```

The results from the second grid search are almost identical to that of the first grid search. However, in the second case, `{'C': 0.05, 'penalty': 'l1', 'solver': 'saga'}` has been identified as the best parameter combination.

|      | Train    | Validate | Test     |
| ---- | -------- | -------- | -------- |
| TNR  | 0.793568 | 0.789942 | 0.798166 |
| TPR  | 0.674301 | 0.648248 | 0.660560 |
| bACC | 0.733935 | 0.719095 | 0.729363 |
| ROC  | 0.788165 | 0.770922 | 0.782781 |
| REC  | 0.674301 | 0.648248 | 0.660560 |
| PRE  | 0.293162 | 0.281451 | 0.293582 |
| AP   | 0.447296 | 0.416163 | 0.441021 |
: Performance metrics of Logistic Regression with L1 regularization

![Confusion Matrix of Logistic Regression with L1 regularization](../figures/6_2_Conf_Mat_2.png){width=45%}

## Statistical Result
Finally, we used the `sm.Logit(y, X)` and `summary()` functions to summarise the performance of the Logistic Regression using raw data. Some features showed very promising predictive power, such as the economic indicators, marital and education.

```python
freq_features = ["job", "marital", "education", "default", "housing", "loan"]
freq_imputer = ColumnTransformer([
    ("freq_imputer", SimpleImputer(missing_values=-1, strategy="most_frequent"), freq_features)
], remainder="passthrough")
# Select "job", "marital", "education"
cat_features = [0,1,2]
# Select "age", "campaign", "pdays", "previous", "emp.var.rate", 
# "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"
num_features = [5,10,11,12,14,15,16,17,18]
hot_scaler = ColumnTransformer([
    ("one_hot_encoder", OneHotEncoder(drop="first"), cat_features),
    ("scaler", StandardScaler(), num_features)
], remainder="passthrough")
freq_transformer = make_pipeline(FunctionTransformer(
    dftransform, kw_args={"drop": ["duration", "y"]}), freq_imputer, hot_scaler)
X = freq_transformer.fit_transform(bank_mkt)
X = np.array(X, dtype=float)
y = bank_mkt["y"].astype("int").to_numpy()
logit_model = sm.Logit(y, X)
result = logit_model.fit(maxiter=1000)
```

|      |     coef | std err |       z | P>\|z\| | [0.025 | 0.975] |
| :--- | -------: | ------: | ------: | ------: | -----: | -----: |
| x1   |  -0.2317 |   0.066 |  -3.507 |       0 | -0.361 | -0.102 |
| x2   |  -0.1243 |   0.106 |  -1.173 |   0.241 | -0.332 |  0.083 |
| x3   |  -0.0808 |   0.126 |  -0.639 |   0.523 | -0.328 |  0.167 |
| x4   |  -0.0922 |   0.074 |  -1.243 |   0.214 | -0.238 |  0.053 |
| x5   |   0.3351 |   0.092 |   3.631 |       0 |  0.154 |  0.516 |
| x6   |   -0.077 |     0.1 |  -0.766 |   0.443 | -0.274 |   0.12 |
| x7   |  -0.1837 |   0.073 |  -2.501 |   0.012 | -0.328 |  -0.04 |
| x8   |   0.2809 |   0.096 |   2.928 |   0.003 |  0.093 |  0.469 |
| x9   |  -0.0133 |   0.061 |  -0.219 |   0.827 | -0.133 |  0.106 |
| x10  |  -0.0062 |    0.11 |  -0.057 |   0.955 | -0.221 |  0.209 |
| x11  |   0.0278 |   0.059 |   0.471 |   0.638 | -0.088 |  0.144 |
| x12  |   0.1051 |   0.067 |   1.567 |   0.117 | -0.026 |  0.237 |
| x13  |  -1.0673 |   0.167 |  -6.401 |       0 | -1.394 |  -0.74 |
| x14  |  -0.9793 |   0.172 |  -5.708 |       0 | -1.316 | -0.643 |
| x15  |   -1.091 |   0.158 |  -6.895 |       0 | -1.401 | -0.781 |
| x16  |  -1.0184 |    0.15 |    -6.8 |       0 | -1.312 | -0.725 |
| x17  |  -0.9735 |   0.158 |  -6.151 |       0 | -1.284 | -0.663 |
| x18  |  -0.8945 |   0.148 |   -6.03 |       0 | -1.185 | -0.604 |
| x19  |  -0.0124 |   0.018 |  -0.703 |   0.482 | -0.047 |  0.022 |
| x20  |  -0.1209 |   0.026 |  -4.706 |       0 | -0.171 | -0.071 |
| x21  |  -0.4997 |   0.026 |  -19.23 |       0 | -0.551 | -0.449 |
| x22  |  -0.0074 |   0.025 |  -0.298 |   0.766 | -0.056 |  0.042 |
| x23  |  -1.3035 |   0.099 | -13.154 |       0 | -1.498 | -1.109 |
| x24  |   0.6383 |   0.057 |  11.289 |       0 |  0.527 |  0.749 |
| x25  |   0.1881 |   0.026 |   7.319 |       0 |  0.138 |  0.238 |
| x26  |   0.4878 |   0.163 |   2.992 |   0.003 |  0.168 |  0.807 |
| x27  |  -0.2901 |   0.109 |  -2.652 |   0.008 | -0.505 | -0.076 |
| x28  | -18.7032 |   30600 |  -0.001 |       1 | -60100 |  60000 |
| x29  |  -0.0287 |   0.035 |  -0.811 |   0.417 | -0.098 |  0.041 |
| x30  |   0.0009 |   0.002 |   0.459 |   0.646 | -0.003 |  0.005 |
| x31  |   -1.018 |    0.06 | -16.887 |       0 | -1.136 |   -0.9 |
| x32  |   -0.053 |   0.011 |   -4.63 |       0 | -0.075 | -0.031 |
| x33  |   0.0432 |   0.012 |   3.507 |       0 |  0.019 |  0.067 |
| x34  |   1.0794 |   0.055 |  19.771 |       0 |  0.972 |  1.186 |
: Statistical result of Logistic Regression
