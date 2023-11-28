# Databricks notebook source
# MAGIC %md
# MAGIC # AutoML tools: Hyperparameter Optimization with Optuna

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook we will be using Optuna for **hyperparameter optimization** in machine learning. Hyperparameter optimization is a critical step in improving the performance of machine learning models. Optuna provides an efficient and automated way to search for the best hyperparameters.
# MAGIC
# MAGIC Before we dive into the specifics of Optuna, let's take a moment to understand what **hyperparameters** are. Hyperparameters are the parameters of a machine learning model that are **not learned** from the data during training. They are **set prior** to training and can have a significant influence on a model's performance and generalization ability. Examples of hyperparameters include the learning rate of an optimizer, the number of hidden layers in a neural network, and the regularization strength in a regression model.
# MAGIC
# MAGIC Selecting appropriate hyperparameters is a crucial aspect of developing effective machine learning models. Poorly chosen hyperparameters can lead to bad performance, including overfitting or underfitting.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's explore Optuna in greater detail to better understand its features and functionalities.

# COMMAND ----------

pip install -q optuna

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
import optuna

# COMMAND ----------

data = pd.read_csv('../../Data/Boston.csv')

X = data.iloc[:, 1:14]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define the objective function

# COMMAND ----------

# MAGIC %md
# MAGIC As we found out before, according to LazyRegressor, the best performing model without hyperparameter tuning is GradientBoostingRegressor with **0.79 r2-score** (see notebook 'AutoML tools: LazyPredict & PyCaret'). Let's optimize this model with Optuna.
# MAGIC
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
# MAGIC #### Create and run the study

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
  min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 70, 10)
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

# COMMAND ----------

# MAGIC %md
# MAGIC Now it's your turn to put what you've learned about the Optuna library into practice! You will try to optimize the model hyperparameters for a classification problem. Select one of the best untuned models based on the results of LazyPredict, create an objective function and run the study. Good luck!

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

# Choose one of the best untuned model based on lazypredict results (see notebook AutoML tools: LazyPredict & PyCaret)
# Find documentation for this model and check which hyperparameters you can tune

# Define objective function
def objective(trial):
  ...
# Note: use classification score function!

# create a new study
study = ...

# optimize an objective function
...

# print the results
...

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
