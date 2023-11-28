# Databricks notebook source
# MAGIC %md
# MAGIC # AutoML tools: LazyPredict & PyCaret

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will explore two powerful AutoML libraries: **LazyPredict** and **PyCaret**. These libraries provide user-friendly interfaces for automating various steps in the machine learning workflow, making it easier for both beginners and experienced data scientists to build and evaluate machine learning models. 
# MAGIC
# MAGIC We will be using these tools for regression (Boston dataset) and classification (Titanic dataset) problems. We will compare their features and limitations. 
# MAGIC
# MAGIC First, we install AutoML libraries.

# COMMAND ----------

pip install -q pycaret lazypredict

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
# MAGIC LazyPredict is an open-source Python library which applies various machine learning models on a dataset and compares their performances. It supports regression and classification problems. 
# MAGIC
# MAGIC LazyPredict is a very simple tool **without hyperparameter tuning**.
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
# MAGIC * Gradient Boosting Regressor
# MAGIC * Bagging Regressor
# MAGIC * Random Forest Regressor
# MAGIC * XGB Regressor 
# MAGIC * Extra Trees Regressor. 
# MAGIC
# MAGIC LazyPredict provides an easy way to see which models work better, so we can focus on them, tune hyperparameters etc.
# MAGIC
# MAGIC The disadvantage is that LazyPredict doesn't give an opportunity to export the best model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression with PyCaret

# COMMAND ----------

# MAGIC %md
# MAGIC PyCaret is an open-source, low-code machine learning Python library, Python wrapper around machine learning libraries and frameworks, such as scikit-learn, XGBoost, LightGBM, CatBoost, and a few more. It was inspired by the emerging role of citizen data scientists, individuals who are not necessarily trained in data science or analytics but have the skills and tools to work with data and extract insights.
# MAGIC
# MAGIC PyCaret supports regression, classification and clustering problems, speeds up experiments and is integrated with BI.
# MAGIC
# MAGIC In this part of the notebook we will explore some of the key features of PyCaret.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's import regression module and setup an experiment. 
# MAGIC
# MAGIC Note: PyCaret can automatically handle common preprocessing tasks, such as handling missing values, feature scaling, and categorical encoding, so we don't need to worry about it.

# COMMAND ----------

from pycaret.regression import *
 
s = setup(boston_df, target = 'target')

# COMMAND ----------

# MAGIC %md
# MAGIC Now that the data is preprocessed, we can use compare_models() function, which trains and evaluates the performance of all the estimators.

# COMMAND ----------

best = compare_models()

# COMMAND ----------

# MAGIC %md
# MAGIC With PyCaret we got very similar list of best regressors.

# COMMAND ----------

# MAGIC %md
# MAGIC ####Optimization
# MAGIC
# MAGIC PyCaret makes it easy to tune hyperparameters of the selected model using the tune_model() function. 
# MAGIC
# MAGIC You can increase the number of iterations (n_iter parameter) depending on how much time and resouces you have. By default, it is set to 10.
# MAGIC
# MAGIC You can also choose which metric to optimize for (optimize parameter). By default, it is set to R2 for regression problem.

# COMMAND ----------

tuned_model = tune_model(best, n_iter = 10, optimize='MAE')

# COMMAND ----------

# MAGIC %md
# MAGIC More advanced features: 
# MAGIC - you can customize the search space (define the search space and pass it to custom_grid parameter)
# MAGIC - you can change the search algorithm. By default, RandomGridSearch is used, but you can change it by setting search_library and search_algorithm parameters
# MAGIC - you can get access to the tuner object. Normally, tune_model only returns the best model. The sample code below shows how it can be done:

# COMMAND ----------

#tuned_model, tuner = tune_model(dt, return_tuner=True)
#print(tuner)

# COMMAND ----------

# MAGIC %md
# MAGIC We can look how hyperparameters have changed:

# COMMAND ----------

# default model
print(best)

# tuned model
print(tuned_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Sometimes tune_model doesn't improve the default model or even gives worse result. If we play around in the notebook where we can choose the best option manually, it's fine. But if we run a python script where we first create models and then tune them, and use the tuned model after, it can be a problem. 
# MAGIC
# MAGIC To solve this, we can set **choose_better** parameter to True, so the best model (default or tuned) will be chosen automatically:

# COMMAND ----------

#tuned_model = tune_model(best, n_iter = 10, optimize='MAE', choose_better=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Analysis
# MAGIC Note that we can easily see the hyperparameters of the model and the whole pipeline, in contrast to LazyPredict library. We also have many other various visualizations provided by the evaluate_model() function.

# COMMAND ----------

evaluate_model(best)

# COMMAND ----------

interpret_model(best)

# COMMAND ----------

# MAGIC %md
# MAGIC *There are many other analyzing tools implemented in PyCaret such as morris sensitivity analysis, reason plot, dashboard etc. You can read more here: https://pycaret.gitbook.io/docs/get-started/functions/analyze.*

# COMMAND ----------

# MAGIC %md
# MAGIC ####Deployment
# MAGIC Let us demonstrate some useful functions:
# MAGIC
# MAGIC - predict_model()
# MAGIC
# MAGIC You can pass to the parameter **data** some new, unseen dataset. In the example below we didn't specify this parameter, so the predictions are made for the holdout set:

# COMMAND ----------

predict_model(tuned_model)

# COMMAND ----------

# MAGIC %md
# MAGIC - finalize_model()
# MAGIC
# MAGIC Refits on the entire dataset including the hold-out set.

# COMMAND ----------

finalize_model(tuned_model)

# COMMAND ----------

# MAGIC %md
# MAGIC - save_model()
# MAGIC
# MAGIC Saves the model as a file in the working directory

# COMMAND ----------

save_model(tuned_model, 'my_best_model')

# COMMAND ----------

# MAGIC %md
# MAGIC - load_model()
# MAGIC
# MAGIC Loads a previosly saved model

# COMMAND ----------

load_model('my_best_model')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Your turn!

# COMMAND ----------

# MAGIC %md
# MAGIC Now, it's time to take your newly acquired knowledge and skills to the next level by trying these powerful AutoML libraries for classification problem.

# COMMAND ----------

# Task: Import titanic.csv dataset

titanic_df = pd.read_csv('../../../../Data/titanic.csv')

# COMMAND ----------

X = titanic_df[['Sex', 'Embarked', 'Pclass', 'Age', 'Survived']]
y = titanic_df[['Survived']]

# COMMAND ----------

X

# COMMAND ----------

y

# COMMAND ----------

# Task: split the dataset into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification with LazyPredict

# COMMAND ----------

# MAGIC %md
# MAGIC *Previously, we used LazyPredict for a regression problem. Now, since you have a classification task, it's recommended to go through the documentation to address the following task: https://lazypredict.readthedocs.io/en/latest/usage.html#classification.*

# COMMAND ----------

# Task: compare different classification models on titanic dataset with LazyClassifier

# Your code here...
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models


# Think how would you interpret the results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification with PyCaret

# COMMAND ----------

# MAGIC %md
# MAGIC *For this new challenge, we encourage you to consult the PyCaret library's documentation to effectively handle the following task: https://pycaret.gitbook.io/docs/get-started/quickstart#classification.*

# COMMAND ----------

# Task: Initialize the environment

from pycaret.classification import *

df_merged = X.merge(y)
print(df_merged)

s = setup(df_merged, target = 'Survived')

# COMMAND ----------

# Task: Compare models

best = compare_models()

# COMMAND ----------

# Task: Optimize the best default model. Set parameters in such a way that the function will return the most efficient model among the default and tuned models.

tuned_model = tune_model(best, n_iter = 10, optimize='Accuracy', choose_better=True)

# COMMAND ----------

# Task: plot confusion matrix

plot_model(tuned_model, plot = 'confusion_matrix', plot_kwargs = {'percent' : True})

# What does the confusion matrix tell us? 

# COMMAND ----------

# Task: get visualization of the pipeline. Hint: use evaluate_model()

evaluate_model(best)

# What is the most important feature? 
# Task: Let's take a look at survival rate by sex. Hint: use seaborn barplot() function. Don't forget to import seaborn!
import seaborn as sns

sns.barplot(x='Sex', y='Survived', data=df_merged)

# What conclusion can we make?

# COMMAND ----------

# Task: save the model as 'my_best_classifier'

save_model(tuned_model, 'my_best_classifier')

# COMMAND ----------

# MAGIC %md
# MAGIC Congratulations! You've completed the study notebook on automating machine learning workflows with PyCaret and LazyPredict. By automating repetitive tasks, these libraries enable us to iterate faster, experiment with various algorithms, and gain valuable insights from our data more efficiently.
# MAGIC
# MAGIC While we explored a wide range of capabilities offered by these libraries, it's essential to note that we haven't covered every single function and feature they provide. As you continue your journey in machine learning, we encourage you to dive deeper into the documentation of both libraries to discover their full range of capabilities.
# MAGIC
# MAGIC **Documentation:**
# MAGIC
# MAGIC - LazyPredict: https://lazypredict.readthedocs.io/en/latest/
# MAGIC
# MAGIC - PyCaret: https://pycaret.gitbook.io/docs/get-started/functions
