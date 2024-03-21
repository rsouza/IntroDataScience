# Databricks notebook source
# MAGIC %md
# MAGIC # AutoML tools: TPOT

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will explore how to use [**TPOT**](https://epistasislab.github.io/tpot/) to automatically optimize machine learning pipelines.
# MAGIC
# MAGIC TPOT, which stands for **Tree-based Pipeline Optimization Tool**, is an open-source AutoML library in Python. It utilizes **genetic programming** to automate the process of feature engineering, model selection, and hyperparameter tuning. TPOT generates and evaluates a population of pipelines, evolving them over generations to identify the most effective combination of data preprocessing steps and machine learning models.
# MAGIC
# MAGIC Before we preceed, let's install TPOT library.

# COMMAND ----------

!pip install -q tpot

# COMMAND ----------

# MAGIC %md
# MAGIC Let's start by importing the necessary libraries and loading the Boston dataset for regression.

# COMMAND ----------

import numpy as np
import pandas as pd
import tpot
from tpot import TPOTRegressor, TPOTClassifier
from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression with TPOT

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the Boston dataset for the regression.

# COMMAND ----------

# Load Boston dataset
boston_df = pd.read_csv("../../../../Data/Boston.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC It is important to preprocess your data before using TPOT. We need to take care of missing values and categorical variables. Let's look at the summary of our DataFrame.

# COMMAND ----------

boston_df.head()

# COMMAND ----------

boston_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC We can see here that all entries have non-null values and float data type for all columns. It means, the only thing we need to do is splitting the data into training and testing sets.

# COMMAND ----------

X = boston_df.iloc[:, 1:14]
y = boston_df.iloc[:, -1]

# Split the Boston data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# COMMAND ----------

# MAGIC %md
# MAGIC The TPOTRegressor performs an intelligent search over machine learning pipelines that can contain supervised regression models, preprocessors, feature selection techniques, and any other estimator or transformer that follows the scikit-learn API. The TPOTRegressor will also search over the hyperparameters of all objects in the pipeline.
# MAGIC
# MAGIC [TPOT Regressor provides various parameters](https://epistasislab.github.io/tpot/api/#regression) to control the optimization process, including:
# MAGIC
# MAGIC * generations: The number of generations (iterations) for the genetic
# MAGIC optimization process.
# MAGIC * population_size: The number of pipelines to maintain in each generation.
# MAGIC * max_time_mins: The maximum time (in minutes) that TPOT should run for optimization.
# MAGIC * scoring: The performance metric used to evaluate the pipelines (e.g., 'neg_mean_squared_error', 'r2', etc.).
# MAGIC * cv: The number of cross-validation folds to use during pipeline evaluation.
# MAGIC * verbosity: The level of verbosity for output during optimization (higher values provide more details).
# MAGIC
# MAGIC TPOT effectiveness improves with more generations, but the trade-off is longer processing time.
# MAGIC
# MAGIC **How can we control the execution time?**
# MAGIC
# MAGIC We can adjust certain parameters to control TPOT execution time:
# MAGIC
# MAGIC * max_time_mins: overrides the generations parameter, specifying the time TPOT runs.
# MAGIC * max_eval_time_mins: how many minutes TPOT spends evaluating a single pipeline.
# MAGIC * early_stop: determines when TPOT ends optimization if no improvement occurs.
# MAGIC * n_jobs: specifies the number of procedures used in parallel during optimization.
# MAGIC * subsample: fraction of training samples used during optimization.
# MAGIC
# MAGIC
# MAGIC But keep in mind that constraining execution time limits TPOT's ability to explore all potential pipelines thoroughly. Consequently, the model suggested within this timeframe may be not the best fit for the dataset. However, if time is sufficient, TPOT can offer something very close to the best model.
# MAGIC
# MAGIC Now, let's create an instance of TPOTRegressor and let it search for the best regression pipeline on the Boston dataset:

# COMMAND ----------

# Create a TPOTRegressor instance
tpot_reg = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42, max_time_mins=5, scoring="neg_mean_squared_error")

# Fit TPOT on the training data for regression
tpot_reg.fit(X_train.values, y_train)
print("Negative MSE on the test set: ", tpot_reg.score(X_test.values, y_test))

# COMMAND ----------

# MAGIC %md
# MAGIC *Note that the default scoring function for regression is negative mean squared error. It is recommended to use the negative version of mean squared error and related metrics so TPOT will minimize (instead of maximize) the metric. You can also create and use your [custom metric](https://epistasislab.github.io/tpot/using/#scoring-functions).*

# COMMAND ----------

# MAGIC %md
# MAGIC There are 4 main functions we can use:
# MAGIC
# MAGIC * fit - Run the TPOT optimization process on the given training data.
# MAGIC * predict - Use the optimized pipeline to predict the target values for a feature set.
# MAGIC * score - Returns the optimized pipeline's score on the given testing data using the user-specified scoring function.
# MAGIC * export - Export the optimized pipeline as Python code.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the resulting pipeline:

# COMMAND ----------

tpot_reg.fitted_pipeline_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification with TPOT

# COMMAND ----------

# MAGIC %md
# MAGIC Scikit-learn includes some [popular datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html). We will load [one of them](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) to illustrate how [**TPOTClassifier**](https://epistasislab.github.io/tpot/api/#classification) works.

# COMMAND ----------

from sklearn.datasets import load_digits
digits = load_digits(as_frame=True) 
#if as_frame parameter is set to True, the data is a pandas DataFrame
digits.frame

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.8, test_size=0.2)

# COMMAND ----------

tpot_class = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, max_time_mins=5)

tpot_class.fit(X_train.values, y_train)

print("Accuracy on the test set: ", tpot_class.score(X_test.values, y_test))

# COMMAND ----------

# MAGIC %md
# MAGIC *Note that the default scoring function for TPOTClassifier is accuracy. We can also use other built-in function such as 'balanced_accuracy', 'f1', 'precision', 'recall' etc.*
# MAGIC
# MAGIC To export the optimized pipeline as Python code we can use the export function:

# COMMAND ----------

# tpot.export('tpot_class_pipeline.py')

# COMMAND ----------

# MAGIC %md
# MAGIC Let's get predictions for the test set:

# COMMAND ----------

predictions = tpot_class.fitted_pipeline_.predict(X_test.values)
predictions

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we used TPOT, a powerful AutoML library, to automate the process of optimizing machine learning pipelines for both regression and classification tasks. We applied TPOT on two different datasets: the Boston dataset for regression and the Digits dataset for classification.
# MAGIC
# MAGIC Using TPOT, we significantly reduced the manual effort of hyperparameter tuning and pipeline construction while achieving competitive performance in both tasks. TPOT is a valuable tool for automating the machine learning workflow and is worth exploring further for various datasets and tasks.
# MAGIC
# MAGIC We highly recommend delving deeper into the documentation of TPOT to gain a comprehensive understanding of its functionalities and capabilities.
# MAGIC
# MAGIC **Documentation:**
# MAGIC
# MAGIC http://epistasislab.github.io/tpot/
# MAGIC
# MAGIC
# MAGIC
# MAGIC Happy automating!
