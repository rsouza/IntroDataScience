# Databricks notebook source
!pip install --upgrade scikit-learn
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Supervised Learning Workflow

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("../../../../Data/data_titanic/train.csv")
train.Pclass = train.Pclass.astype(float)  # to avoid DataConversionWarning

# COMMAND ----------

train = train.dropna(axis=0)
X_train, X_test, y_train, y_test = train_test_split(
    train[["Pclass", "Age", "Sex", "Embarked"]],
    train["Survived"],
    test_size=0.2,
    random_state=42,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Tree-based Models & Hyperparameter Tuning
# MAGIC In the previous notebook we constructed our first pipeline:
# MAGIC ```
# MAGIC entire_pipeline = Pipeline([('feature_engineering', feature_engineering), ('dummy', DummyClassifier(strategy="most_frequent"))])
# MAGIC ```
# MAGIC Hold your constructed pipeline firmly! The only thing that we need to do now is to replace [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) with a proper learning model.   
# MAGIC We can start with a decision tree.

# COMMAND ----------

feature_engineering = ColumnTransformer(
    [
        ("numerical_scaler", preprocessing.MinMaxScaler(), ["Pclass", "Age"]),
        ("ohe", preprocessing.OneHotEncoder(sparse_output=False), ["Sex", "Embarked"]),
    ],
    remainder="passthrough",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fitting a Learning Model – Decision Tree

# COMMAND ----------

# TASK 1A: Reuse your composite and instead of a dummy, fit a decision tree with default parameters.
# Store the result as dt_pipeline.

# Train the pipeline

# COMMAND ----------

# TASK 1B: Let the pipeline predict for the training set. 
# Store the result as y_pred_TRAIN_DT.
# Also, display accuracy.

# COMMAND ----------

# TASK 1C: Let the pipeline predict for the holdout set. 
# Store the result as y_pred_HOLDOUT_DT.
# Also, display accuracy.

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at the accuracy on training and holdout set, what can you infer about over model? Will it generalize well?

# COMMAND ----------

# OPTIONAL TASK 2: Do the same steps with RandomForest with default parameters. 
# Does the RandomForest display similar results as decision tree? If not, why?

# Reuse your composite and fit a random forest with default parameters.
# Store the result as rf_pipeline.

# Train the pipeline

#Predict and show accuracy TRAIN

#Predict and show accuracy HOLDOUT

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tuning Hyperparameters of our Decision Tree
# MAGIC Time to improve the performance of our learning model by finding its optimal set of hyperparameters.  
# MAGIC We start by examining **which hyperparameters are available** in our decision tree pipeline.

# COMMAND ----------

dt_pipeline.get_params()

# COMMAND ----------

# MAGIC %md
# MAGIC We would like to tune `max_depth` and `min_samples_split`.  
# MAGIC Notice that to access them, we also need to navigate within the composite and call them as **`decision_tree`**`__max_depth`.  

# COMMAND ----------

# TASK 3: Define a grid through which we should search. 
# Tune parameters: max_depth and min_samples_split.
# The values which you pick as parameters are up to you. You can think about them intuitively.

# COMMAND ----------

from sklearn import tree
from sklearn.model_selection import GridSearchCV

# Model
dt_pipeline

# Searching strategy, providing grid
tuning = GridSearchCV(dt_pipeline, param_grid)

# Train
tuning.fit(X_train, y_train)

# COMMAND ----------

# Let's get the best parameters
best_par = tuning.best_params_
print(best_par)

# COMMAND ----------

# MAGIC %md
# MAGIC If you want to have a more detailed look at the result from the grid search you can use the `cv_results_` attribute.
# MAGIC The dict is easily transformed to a pandas DataFrame.

# COMMAND ----------

# Let's check out the full Grid Search results
# We sort the dataframe according to the rank and have a look at the top 10 models
gs_result = pd.DataFrame(tuning.cv_results_)
gs_result.sort_values("rank_test_score").head(10)

# COMMAND ----------

# TASK 4A: Use the best setting of the two hyperparameters and fit a optimized decision tree. 
# Hint: Reuse the pipeline and when declaring it, specify the params.

# Store it as dt_pipeline_tuned.

# Train

# COMMAND ----------

# TASK 4B: Display accuracy on the training set of the optimized decision tree.

# COMMAND ----------

# TASK 4C: Display accuracy on the holdout set of the optimized decision tree.

# COMMAND ----------

# MAGIC %md
# MAGIC Does the optimized decision tree perform better then the one with default parameters?
# MAGIC
# MAGIC The best model can also be retrieved directly from the result of the grid search, if the parameter `refit=True` is used.
# MAGIC By default the value of this parameter is `True` so instead of manually retraining we could eiter use the attribute `best_estimator_` to retrieve the model or make predictions by using the 
# MAGIC [`predict()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV.predict)
# MAGIC straight away.

# COMMAND ----------

# retrieve the best model from the Grid Search object
dt_tuning = tuning.best_estimator_
print(metrics.accuracy_score(y_test, dt_tuning.predict(X_test)))

# directly predict using the Grid Search object
print(metrics.accuracy_score(y_test, tuning.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional Advanced TASK: Tuning Random Forest
# MAGIC When you are tuning a more complex model, it is good practice to search available literature on which hyperparameters should be tuned. Below I have predefined some. You can **play around with the grid**, for example expand or narrow it. Keep in mind that as our feature set is extremely limited, its hard for hyperparameter tuning to arrive at something meaningful.

# COMMAND ----------

# OPTIONAL TASK 5
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define a pipeline
rf_pipeline = Pipeline([('feature_engineering', feature_engineering), ('random_forest', RandomForestClassifier())])

# Create the parameter grid based on the results of random search 
param_grid_rf = {
    'random_forest__bootstrap': [True, False],
    'random_forest__max_depth': [3, 5, 10, 15],
    'random_forest__max_features': [2, 3],
    'random_forest__min_samples_leaf': [3, 4, 5],
    'random_forest__min_samples_split': [5, 8, 10, 12],
    'random_forest__n_estimators': [5, 10, 15, 20, 25]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf_pipeline, 
                           param_grid = param_grid_rf, 
                           cv = 3, 
                           n_jobs = -1, 
                           verbose = 2)

# Searching strategy, providing grid
tuning_rf = GridSearchCV(rf_pipeline, param_grid_rf)

# Train
tuning_rf.fit(X_train, y_train)

# Cross-validated score (more robust than holdout set most likely)
print(tuning_rf.best_score_)
print(tuning_rf.best_params_)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional Advanced TASK: Check Kaggle competitions and join one of them!  
# MAGIC https://www.kaggle.com/
