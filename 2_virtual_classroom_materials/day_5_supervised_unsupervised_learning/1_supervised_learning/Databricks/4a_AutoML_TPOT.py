# Databricks notebook source
# MAGIC %md
# MAGIC # AutoML tools: TPOT

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will explore how to use **TPOT** to automatically optimize machine learning pipelines.
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

# Load Boston dataset
boston_df = pd.read_csv("../../../../Data/Boston.csv")

X = boston_df.iloc[:, 1:14]
y = boston_df.iloc[:, -1]

# Split the Boston data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# COMMAND ----------

# MAGIC %md
# MAGIC The TPOTRegressor performs an intelligent search over machine learning pipelines that can contain supervised regression models, preprocessors, feature selection techniques, and any other estimator or transformer that follows the scikit-learn API. The TPOTRegressor will also search over the hyperparameters of all objects in the pipeline.
# MAGIC
# MAGIC TPOT Regressor provides various parameters to control the optimization process, including:
# MAGIC
# MAGIC * generations: The number of generations (iterations) for the genetic
# MAGIC optimization process.
# MAGIC * population_size: The number of pipelines to maintain in each generation.
# MAGIC * max_time_mins: The maximum time (in minutes) that TPOT should run for optimization.
# MAGIC * scoring: The performance metric used to evaluate the pipelines (e.g., 'neg_mean_squared_error', 'r2', etc.).
# MAGIC * cv: The number of cross-validation folds to use during pipeline evaluation.
# MAGIC * verbosity: The level of verbosity for output during optimization (higher values provide more details).
# MAGIC
# MAGIC Now, let's create an instance of TPOTRegressor and let it search for the best regression pipeline on the Boston dataset:

# COMMAND ----------

# Create a TPOTRegressor instance
tpot_reg = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)

# Fit TPOT on the training data for regression
tpot_reg.fit(X_train, y_train)

# COMMAND ----------

# Evaluate the best regression pipeline on the test set
best_pipeline_reg = tpot_reg.fitted_pipeline_
test_score_reg = best_pipeline_reg.score(X_test, y_test)
print(f'Test Set R^2 Score (Regression): {test_score_reg:.3f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification with TPOT

# COMMAND ----------

# MAGIC %md
# MAGIC *You may use the documentation as a guide:*
# MAGIC
# MAGIC *http://epistasislab.github.io/tpot/api/*

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

# Task: Create a TPOTClassifier instance
...

# Task: Fit TPOT on the training data for classification
...

# COMMAND ----------

# Task: Evaluate the best classification pipeline on the test set
...

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we used TPOT, a powerful AutoML library, to automate the process of optimizing machine learning pipelines for both regression and classification tasks. We applied TPOT on two different datasets: the Boston dataset for regression and the Titanic dataset for classification.
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
