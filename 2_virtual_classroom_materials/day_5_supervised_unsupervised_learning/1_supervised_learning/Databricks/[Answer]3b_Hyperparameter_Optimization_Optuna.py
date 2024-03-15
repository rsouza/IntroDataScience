# Databricks notebook source
# MAGIC %md
# MAGIC # AutoML tools: Hyperparameter Optimization with Optuna

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook we will be using [Optuna](https://optuna.readthedocs.io/en/stable/index.html) for **hyperparameter optimization** in machine learning. 
# MAGIC Hyperparameter optimization is a critical step in improving the performance of machine learning models.
# MAGIC [Optuna](https://optuna.readthedocs.io/en/stable/index.html)
# MAGIC provides an efficient and automated way to search for the best hyperparameters.
# MAGIC
# MAGIC Before we dive into the specifics of [Optuna](https://optuna.readthedocs.io/en/stable/index.html), 
# MAGIC let's take a moment to understand what **hyperparameters** are. 
# MAGIC Hyperparameters are the parameters of a machine learning model that are **not learned** from the data during training. They are **set prior** to training and can have a significant influence on a model's performance and generalization ability.
# MAGIC Examples of hyperparameters include the learning rate of an optimizer, the number of hidden layers in a neural network, and the regularization strength in a regression model.
# MAGIC
# MAGIC Selecting appropriate hyperparameters is a crucial aspect of developing effective machine learning models. Poorly chosen hyperparameters can lead to bad performance, including overfitting or underfitting.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's explore [Optuna](https://optuna.readthedocs.io/en/stable/index.html) in greater detail to better understand its features and functionalities.

# COMMAND ----------

pip install -q optuna

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
import optuna

# COMMAND ----------

data = pd.read_csv('../../../../Data/Boston.csv')

X = data.iloc[:, 1:14]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a baseline
# MAGIC
# MAGIC For this regression task we will optimize the hyperparameters of a [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).
# MAGIC The performance of the models will be compared by the [**\\(r^2\\)-score**](https://en.wikipedia.org/wiki/Coefficient_of_determination).
# MAGIC
# MAGIC Let's start out by fitting a
# MAGIC [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
# MAGIC with default parameters.

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

baseline_regressor = GradientBoostingRegressor(random_state=0)

baseline_regressor.fit(X_train, y_train)
baseline_r2 = baseline_regressor.score(X_test, y_test)

print(f"The baseline r2-score is {baseline_r2:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC We find that on the test set the \\(r^2\\)-score for a
# MAGIC [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
# MAGIC without any hyperparameter tuning is about 0.79.
# MAGIC In the following sections we will improve this score by optimizing the model with
# MAGIC [Optuna](https://optuna.readthedocs.io/en/stable/index.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the objective function
# MAGIC We start with **defining objective function**. The objective function is a crucial component of hyperparameter optimization. It defines the metric you want to optimize (e.g., accuracy, loss). This function takes hyperparameters as input, builds and trains a model, and evaluates its performance on a validation set.
# MAGIC
# MAGIC Let's take a look at GradientBoostingRegressor documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html.
# MAGIC
# MAGIC As we can see, GradientBoostingRegressor has many parameters. Which of them should we optimize?
# MAGIC
# MAGIC The overall parameters can be divided into 3 categories:
# MAGIC
# MAGIC - **Tree-Specific Parameters** min_samples_split, min_samples_leaf, max_leaf_nodes etc.
# MAGIC - **Boosting Parameters**: learning_rate, n_estimators, subsample
# MAGIC - **Miscellaneous Parameters**: loss, random_state etc.
# MAGIC
# MAGIC We will tune only first two types of parameters. *Let's follow the general approach for parameter tuning, explained in this article: https://luminousdata.wordpress.com/2017/07/27/complete-guide-to-parameter-tuning-in-gradient-boosting-gbm-in-python/.*
# MAGIC
# MAGIC First, we take a default **learning rate** (0.1). Now we should determine the **optimum number of trees** (n_estimators) for this learning rate.

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Objective function
def objective_1(trial):

  # Define hyperparameters to optimize
  # Suggest the number of trees in range [10, 300]
  n_estimators = trial.suggest_int('n_estimators', 10, 300)

  model = GradientBoostingRegressor(
        n_estimators = n_estimators,
        random_state = 42
    )

  # Train the model
  model.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test)

  # Calculate r2
  r2 = r2_score(y_test, y_pred)
  return r2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create and run the study

# COMMAND ----------

# Create a new study (set of trials)
study = optuna.create_study(direction='maximize')

# Optimize an objective function
study.optimize(objective_1, n_trials=50)

# Print the results
print('Study #1')
# Attribute 'trials' returns the list of all trials
print('Number of finished trials:', len(study.trials))
# Attribute 'best_trial' returns the best trial in the study
print('Best trial:')
trial = study.best_trial
# 'value' returns the r2-score of the best trial in the study
print('Value:', trial.value)
print('Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC Our study showed **the optimum number of trees**.
# MAGIC
# MAGIC Now we will use this number of trees in our model and tune tree-specific parameters. We should choose the order of tuning variables wisely, i.e. start with the ones that have a bigger effect on the outcome. For example, we need to focus on variables max_depth and min_samples_split first, as they have a strong impact.
# MAGIC
# MAGIC Let's tune **max_depth** and **min_samples_split**.

# COMMAND ----------

def objective_2(trial):

  # Define hyperparameters to optimize
  max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
  min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1)

  model = GradientBoostingRegressor(
        n_estimators = 178,
        max_depth = max_depth,
        min_samples_split = min_samples_split,
        random_state = 42
    )

  # Train the model
  model.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test)

  # Calculate accuracy
  r2 = r2_score(y_test, y_pred)
  return r2

study = optuna.create_study(direction='maximize')

study.optimize(objective_2, n_trials=100)

print('Study #2')
print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('Value:', trial.value)
print('Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC We got the best values for max_depth and min_samples_split. At this point, we can notice that there is a big impovement in r2-score compared to the untuned model.
# MAGIC
# MAGIC Now, let's keep max_depth in our model and tune **min_samples_split** and **min_samples_leaf** together.

# COMMAND ----------

def objective_3(trial):

  # Define hyperparameters to optimize
  min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 71, step=10)
  min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1)

  model = GradientBoostingRegressor(
        n_estimators=178,
        max_depth=24,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

  # Train the model
  model.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test)

  # Calculate accuracy
  r2 = r2_score(y_test, y_pred)
  return r2

study = optuna.create_study(direction='maximize')

study.optimize(objective_3, n_trials=50)

print('Study #3')
print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('Value:', trial.value)
print('Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC We have the last tree-specific parameter we need to tune - **max_features**. We will try values from 1 to 13 in steps of 2.

# COMMAND ----------

def objective_4(trial):

  # Define hyperparameters to optimize
  max_features = trial.suggest_int('max_features', 1, 13, 2)

  model = GradientBoostingRegressor(
        n_estimators=178,
        max_depth=24,
        min_samples_split=0.31220765553286495,
        min_samples_leaf=1,
        max_features = max_features,
        random_state=42
    )

  # Train the model
  model.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test)

  # Calculate accuracy
  r2 = r2_score(y_test, y_pred)
  return r2

study = optuna.create_study(direction='maximize')

study.optimize(objective_4, n_trials=50)

print('Study #4')
print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('Value:', trial.value)
print('Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will tune boosting parameter **subsample**.

# COMMAND ----------

def objective_5(trial):
  # Define hyperparameters to optimize
  subsample = trial.suggest_float('subsample', 0.6, 1, step=0.05)

  model = GradientBoostingRegressor(
        n_estimators=178,
        max_depth=24,
        min_samples_split=0.31220765553286495,
        min_samples_leaf=1,
        max_features = 11,
        subsample=subsample,
        random_state=42
    )

    # Train the model
  model.fit(X_train, y_train)

    # Make predictions on the test set
  y_pred = model.predict(X_test)

    # Calculate accuracy
  r2 = r2_score(y_test, y_pred)
  return r2

study = optuna.create_study(direction='maximize')

study.optimize(objective_5, n_trials=50)

print('Study #5')
print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('Value:', trial.value)
print('Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC It can be seen, that default value of subsample is optimal.
# MAGIC
# MAGIC Now we will create a final model with all the parameters we tuned.

# COMMAND ----------

final_model = GradientBoostingRegressor(
    n_estimators=178,
    max_depth=24,
    min_samples_split=0.31220765553286495,
    min_samples_leaf=1,
    max_features=11,
    subsample=1,
    random_state=42
)

final_model.fit(X_train, y_train)
final_predictions = final_model.predict(X_test)
final_score = r2_score(y_test, final_predictions)
print('Trained and evaluated the final model using the best hyperparameters.\n')
print('Final model score:', final_score)

# COMMAND ----------

# MAGIC %md
# MAGIC Perfect! Using the Optuna library, we tuned the hyperparameters of the model and got an improvement in the r2-score.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your turn!
# MAGIC Now it's your turn to put what you've learned about the Optuna library into practice! You will try to optimize the model hyperparameters for a classification problem.
# MAGIC
# MAGIC In the code chunk below we load the data for this task.
# MAGIC The goal is to predict if a patient is obese.
# MAGIC There are 4 classes  in the target variable.

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder

# Import obesity_data.csv dataset
obesity_df = pd.read_csv("../../../../Data/obesity_data.csv")

X = obesity_df.iloc[:,:16]
y = obesity_df[['NObeyesdad']]

# Encode the categorical variables
obesity_preprocessing = OneHotEncoder(drop="if_binary")

X = obesity_preprocessing.fit_transform(X)

# COMMAND ----------

obesity_df

# COMMAND ----------

# Task: split the dataset into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC Again we create a baseline with default parameters.
# MAGIC In this task we use a
# MAGIC [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

baseline_classifier = RandomForestClassifier(random_state=0)

baseline_classifier.fit(X_train, y_train)
baseline_accuracy = baseline_classifier.score(X_test, y_test)

print(f"The baseline accuracy-score is {baseline_accuracy:.3f}")

# COMMAND ----------

# Choose one of the best untuned model based on lazypredict results (see notebook AutoML tools: LazyPredict & PyCaret)
# Find documentation for this model and check which hyperparameters you can tune

# Define objective function
def objective(trial):
    # Define hyperparameters to optimize
    # Suggest the number of trees in range [50, 200]
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    # Suggest the function to measure the quality of a split
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    # Suggest the maximum depth of the tree in range [1, 100]
    max_depth = trial.suggest_int("max_depth", 1, 100)
    # Suggest the number of samples required to split an internal node  in range [2, 20]
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    # Create the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=0,
    )

    # Train the model
    model.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    return accuracy


# Note: use classification score function!

# create a new study
study = optuna.create_study(direction='maximize')

# Optimize an objective function
study.optimize(objective, n_trials=50)

# Print the results
print('Study #1')
# Attribute 'trials' returns the list of all trials
print('Number of finished trials:', len(study.trials))
# Attribute 'best_trial' returns the best trial in the study
print('Best trial:')
trial = study.best_trial
# 'value' returns the accuracy score of the best trial in the study
print('Value:', trial.value)
print('Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC Congratulations! :) You finished the notebook about hyperparameter optimization with Optuna.
# MAGIC
# MAGIC This notebook has provided an introduction to the Optuna library and its significance in automating hyperparameter tuning for machine learning models. By utilizing its functionalities, we efficiently tuned the hyperparameters for a regression and classification problems.
# MAGIC
# MAGIC We encourage you to explore the Optuna documentation further.
# MAGIC
# MAGIC **Documentation:**
# MAGIC
# MAGIC https://optuna.readthedocs.io/en/stable/reference/index.html.
# MAGIC
# MAGIC Keep up the excellent work!
