# Databricks notebook source
# MAGIC %md
# MAGIC # AutoML tools: LazyPredict

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will explore a simple AutoML library:
# MAGIC [**LazyPredict**](https://lazypredict.readthedocs.io/en/latest/).
# MAGIC
# MAGIC We will be using these tools for regression (Boston dataset) and classification (Titanic dataset) problems. We will explore their features and limitations. 
# MAGIC
# MAGIC First, we install the
# MAGIC [**LazyPredict**](https://lazypredict.readthedocs.io/en/latest/)
# MAGIC library.

# COMMAND ----------

pip install -q lazypredict

# COMMAND ----------

# You only need to run this cell after installing the optuna package on Databricks
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Then we load the Boston dataset using Pandas.

# COMMAND ----------

import pandas as pd

boston_df = pd.read_csv('../../../../Data/Boston.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC Before using AutoML tools, let's take a quick look at our dataset and its structure:

# COMMAND ----------

boston_df.head()

# COMMAND ----------

boston_df.describe()

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = boston_df.iloc[:, 1:14]
y = boston_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression with LazyPredict

# COMMAND ----------

# MAGIC %md
# MAGIC [LazyPredict](https://lazypredict.readthedocs.io/en/latest/)
# MAGIC is an open-source Python library which applies various machine learning models on a dataset and compares their performances.
# MAGIC It supports regression and classification problems. 
# MAGIC
# MAGIC [LazyPredict](https://lazypredict.readthedocs.io/en/latest/)
# MAGIC is a very simple tool **without hyperparameter tuning**.
# MAGIC
# MAGIC Let's try it out!

# COMMAND ----------

from lazypredict.Supervised import LazyRegressor

reg = LazyRegressor(predictions=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# COMMAND ----------

models

# COMMAND ----------

predictions

# COMMAND ----------

# MAGIC %md
# MAGIC You can also pass to LazyRegressor() additional optional parameters such as the **verbose** flag, which controls the level of output produced during training, and the **custom_metric** parameter, which allows you to specify a custom metric to use for evaluating the model. See example below:

# COMMAND ----------

from sklearn.metrics import mean_absolute_error

reg = LazyRegressor(verbose=0, predictions=True, custom_metric = mean_absolute_error)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# COMMAND ----------

models

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can see our custom metric in the last column.

# COMMAND ----------

# MAGIC %md
# MAGIC We got top-5 models: 
# MAGIC * [Gradient Boosting Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
# MAGIC * [Bagging Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)
# MAGIC * [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
# MAGIC * [XGB Regressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn)
# MAGIC * [Extra Trees Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html). 
# MAGIC
# MAGIC LazyPredict provides an easy way to see which models work better, so we can focus on them, tune hyperparameters etc.
# MAGIC
# MAGIC The disadvantage is that LazyPredict doesn't give an opportunity to export the best model.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Your turn!

# COMMAND ----------

# MAGIC %md
# MAGIC Now, it's time to take your newly acquired knowledge and skills to the next level by trying the LazyPredict library for classification problem.

# COMMAND ----------

# Task: Import titanic.csv dataset

titanic_df = ...

# COMMAND ----------

X = titanic_df[['Sex', 'Embarked', 'Pclass', 'Age', 'Survived']]
y = titanic_df[['Survived']]

# COMMAND ----------

# Task: split the dataset into train and test sets

...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification with LazyPredict

# COMMAND ----------

# MAGIC %md
# MAGIC *Previously, we used LazyPredict for a regression problem. Now, since you have a classification task, it's recommended to go through the documentation to address the following task: https://lazypredict.readthedocs.io/en/latest/usage.html#classification.*

# COMMAND ----------

# Task: compare different classification models on titanic dataset with LazyClassifier

# Your code here...

# Think how would you interpret the results

# COMMAND ----------

# MAGIC %md
# MAGIC Congratulations! You've completed the study notebook on automating machine learning workflows with and LazyPredict. 
# MAGIC By automating repetitive tasks, this library enables us to iterate faster, experiment with various algorithms, and gain valuable insights from our data more efficiently.
# MAGIC
# MAGIC As you continue your journey in machine learning and keep on using this library, we encourage you to dive deeper into the documentation and Github page.
# MAGIC
# MAGIC **Resources:**
# MAGIC - Documentation: https://lazypredict.readthedocs.io/en/latest/
# MAGIC - Github: https://github.com/shankarpandala/lazypredict
