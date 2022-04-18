# Conclusion
```{=latex}
\chapterauthor{Jiawei Li}
```

In conclusion, this project has used `python` and its rich ecosystems to explore the bank marketing dataset, build data pipelines and utilise machine learning classification models including Logistic Regression, Support Vector Machine, Neural Network, Decision Trees and Ensemble Learning. We have also discussed performance evaluation and concluded that Average Precision and ROC AUC are the prefered metrics due to their flexibility on the models’ thresholds and interpretability for imbalanced datasets.

Our exploratory and modelling process shows that it is possible to build a predictive model for improving bank telemarketing results. Using `XGBoost` without client demographic feature achieves the best performance at 0.813 ROC AUC and 0.483 Average Precision on the test set.

There are several reasons why our model can not further improve. The first reason is that the Financial crisis in 2008 and Portugal’s debt crisis in 2010 may alter people’s deposit choices. There is a clear pattern that in 2009 and 2010 clients are much more likely to subscribe the term deposit (5% positive in 2008, 25% positive in 2009, 50% positive in 2010) which may imply that people are changing their investment preferences or the marketing staff have improved their sale skills. The second reason is the lack of data. Data from 2009 and 2010 only contributes to 25% of the whole dataset but has the majority of positive outcomes. By collecting data each year and expand the range of years, the model could achieve better results.
