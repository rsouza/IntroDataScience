# Databricks notebook source
# MAGIC %md
# MAGIC ## Importing

# COMMAND ----------

import pandas as pd

train = pd.read_csv("data_titanic/train.csv")

# COMMAND ----------

train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Brief Exploration

# COMMAND ----------

#Categorical features
train.describe(include = object)

# COMMAND ----------

#Numerical features
train.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's work only with the following for simplicity
# MAGIC Categorical:
# MAGIC - Sex
# MAGIC - Embarked
# MAGIC 
# MAGIC Numerical:
# MAGIC - Survived: *target* 0 = No, 1 = Yes
# MAGIC - Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
# MAGIC - Age: Age in years
# MAGIC  
# MAGIC More detailed info: https://www.kaggle.com/c/titanic

# COMMAND ----------

#Let's keep only the desired columns
train = train[['Sex','Embarked','Pclass', 'Age','Survived']]

# COMMAND ----------

train.shape

# COMMAND ----------

#Check for missing values
train.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC For simplicity, we drop rows with missing values. If you will later experiment with composite transformers, comment out this cell so that you try to include also missing value imputation.

# COMMAND ----------

train = train.dropna(axis=0)

# COMMAND ----------

train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC With our current knowledge, we can try to implement individually various transformers from scikit-learn. Let's not forget to create a holdout set!

# COMMAND ----------

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train[['Pclass', 'Age', 'Sex', 'Embarked']],
                                                    train['Survived'], 
                                                    test_size=0.2, 
                                                    random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical Features
# MAGIC - Pclass
# MAGIC - Age  
# MAGIC Let's just scale these two features using MinMax scaler.

# COMMAND ----------

scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train[['Pclass', 'Age']])
X_train_transformed_numerical = scaler.transform(X_train[['Pclass', 'Age']])
X_test_transformed_numerical = scaler.transform(X_test[['Pclass', 'Age']])

print(X_train_transformed_numerical.shape)
print(X_test_transformed_numerical.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical Features
# MAGIC *   Sex
# MAGIC *   Embarked
# MAGIC 
# MAGIC We can simply one-hot encode these.

# COMMAND ----------

encoder = preprocessing.OneHotEncoder(sparse=False)
encoder.fit(X_train[['Sex', 'Embarked']])
X_train_transformed_categorical = encoder.transform(X_train[['Sex', 'Embarked']])
X_test_transformed_categorical = encoder.transform(X_test[['Sex', 'Embarked']])

print(X_train_transformed_categorical.shape)
print(X_test_transformed_categorical.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## HANDS-ON 1: Baseline Model & Model Evaluation
# MAGIC Time for first exercise! At first, let's put together the transformed numerical and categorical features.

# COMMAND ----------

X_train_transformed = np.concatenate((X_train_transformed_numerical,X_train_transformed_categorical), axis = 1)
X_test_transformed = np.concatenate((X_test_transformed_numerical,X_test_transformed_categorical), axis = 1)

print(X_train_transformed.shape)
print(X_test_transformed.shape)

# COMMAND ----------

# TASK 1: Fit sklearn.DummyClassifier. Then, let the model predict for train (X_train_transformed) and holdout set(X_test_transformed).
# Store the prediction as y_pred_TRAIN_DUMMY (training set) and as y_pred_HOLDOUT_DUMMY (holdout set)

from sklearn.dummy import DummyClassifier


# COMMAND ----------

# OPTIONAL TASK 1: Think about a simple heuristic that can be used as baseline. 
# One possibility is to use gender and for example predict that every men or every woman has survived.
# You can store the result as y_pred_TRAIN_HEURISTIC and as y_pred_HOLDOUT_HEURISTIC.




# COMMAND ----------

# MAGIC %md
# MAGIC Great! We have our first prediction! It is time to evaluate how good our (poor dummy) model is. It is time to use the *sklearn.metrics* module.   
# MAGIC https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

# COMMAND ----------

from sklearn import metrics

#TASK 2A: Display ACCURACY on TRAIN set.

#TASK 2B: Display ACCURACY on HOLDOUT set.

#OPTIONAL TASK 2C: Can you think of better measure than accuracy based on the domain problem? If yes, use it the same way.


# COMMAND ----------

# MAGIC %md
# MAGIC Great, now we would also like to see confusion matrix as it is always a good idea to see visually the quality of our predictions.

# COMMAND ----------

#TASK 3: Display a CONFUSION MATRIX on HOLDOUT set. Hint: do not use plot_confusion_matrix but confusion_matrix only.


# COMMAND ----------

# MAGIC %md
# MAGIC ## HANDS-ON 2: Composite Estimators
# MAGIC Let's nicely wrap our Feature Engineering and model fitting into a nice composite estimator. We will be very simplistic and only use two  
# MAGIC They will not nest into each other at once.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering wrapped into ColumnTransformer
# MAGIC The two feature transformations can be easily wrapped up into a single ColumnTransformer. This will ensure that our Feature Engineering is a **bit more robust and nicely encapsulated**. Refer to the section 6.1.4 of the following link. It will showcase the exact application that we intend to create:
# MAGIC 
# MAGIC https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data

# COMMAND ----------

#TASK 3: Wrap MinMaxScaler and OneHotEncoder into a single ColumnTransformer. The transformers should be applied to according columns only.
#Store the resulting composite as feature_engineering
# Hint: use argument remainder='passthrough'
from sklearn.compose import ColumnTransformer


# COMMAND ----------

# MAGIC %md
# MAGIC ### Predictive Model Wrapped into Pipeline
# MAGIC Let's now wrap together feature engineering with the model into a single Pipeline Composite estimator. Here is a pseudocode:
# MAGIC - entire_pipeline = feature_engineering -> model  
# MAGIC 
# MAGIC Both components are already available. From step above, we can directly reuse the object feature_engineering. As model, we just call new DummyClassifier, just as we did before.

# COMMAND ----------

# TASK 4: Wrap Feature Engineering and Predictive Model (dummy) into a single Pipeline composite estimator. 
# Store the result as entire_pipeline
from sklearn.pipeline import Pipeline


# COMMAND ----------

# TASK: Uncomment the line and try to train the pipeline.
# It should not return an error. 
# Notice that we are using untransformed data again (X_train) as the pipeline contains the transformers.

#entire_pipeline.fit(X = X_train, y = y_train)

# COMMAND ----------

#Predict for training data
y_pred_TRAIN_DUMMY = entire_pipeline.predict(X_train)
#Predict for holdout data
y_pred_HOLDOUT_DUMMY = entire_pipeline.predict(X_test)

#Results should be the same as before
print(metrics.accuracy_score(y_train, y_pred_TRAIN_DUMMY))

#TASK 2B: Display ACCURACY on HOLDOUT set.
print(metrics.accuracy_score(y_test, y_pred_HOLDOUT_DUMMY))

# COMMAND ----------

# MAGIC %md
# MAGIC OPTIONAL TASK:   
# MAGIC A notebook 'nice_pipeline' was made to exemplify some examples of more complex pipelines. Feel free to scroll through it and learn how a process of preparing a complex composite looks like. You can then come back here and try to implement various components. For example, if I would not drop rows with missing values at the beginning of this notebook, constructing a composite would get a bit trickier. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## HANDS-ON 3: Tree-based Models & Hyperparameter Tuning
# MAGIC Hold your constructed Pipeline firmly! The only thing that we need to do now, is to replace the DummyClassifier with a proper learning model. We can start by a decision tree.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fitting Learning Model - Decision Tree

# COMMAND ----------

# TASK 5: Reuse your composite, instead of a dummy, fit a decision tree with default parameters.
# Store the result as dt_pipeline
from sklearn.tree import DecisionTreeClassifier


# Train the pipeline
#dt_pipeline.fit(X = X_train, y = y_train)

# COMMAND ----------

# TASK 5B: Let the pipeline predict for TRAINING set. Store the result as y_pred_TRAIN_DT
# Also, Display accuracy.


# COMMAND ----------

# TASK 5C: Let the pipeline predict for HOLDOUT set. Store the result as y_pred_HOLDOUT_DT
# Also, Display accuracy.


# COMMAND ----------

# MAGIC %md
# MAGIC Looking at the accuracy on training and holdout set, what can you infer about over model? Will it generalize well?

# COMMAND ----------

# OPTIONAL TASK 6: Do the same steps with RandomForest with default parameters. 
# Does the RandomForest display similar results as decision tree? If not, why?
from sklearn.ensemble import RandomForestClassifier


# COMMAND ----------

# MAGIC %md
# MAGIC ### Tuning Hyperparameters of our Decision Tree
# MAGIC Time to improve the performance of our learning model by finding its optimal set of hyperparameters. We start by examining **what hyperparameters are available** in our decision tree pipeline.

# COMMAND ----------

dt_pipeline.get_params()

# COMMAND ----------

# MAGIC %md
# MAGIC We would like to tune max_depth and min_samples_split. Notice that to access them, we also need to navigate within the composite and call them as *decision_tree__max_depth*.

# COMMAND ----------

# TASK 7: Define a grid through which we should search. Tune parameters: max_depth and min_samples_split.
# The values which you pick for parameters are up to you. You can think about them intuitively.


# COMMAND ----------

from sklearn import tree
from sklearn.model_selection import GridSearchCV

#Model
dt_pipeline

#Searching strategy, providing grid
tuning = GridSearchCV(dt_pipeline, param_grid)

#Train
tuning.fit(X_train, y_train)

# COMMAND ----------

#Let's get the best parameters
tuning.best_params_

# COMMAND ----------

# TASK 8: Use the best setting of the two hyperparameters and fit a optimized decision tree. Hint: Reuse the pipeline, just when declaring it, specify the params.
# Store it as dt_pipeline_tuned

# Train
#dt_pipeline_tuned.fit(X_train, y_train)

# COMMAND ----------

# TASK 8B: Display accuracy on TRAINING set of the optimized decision tree.


# COMMAND ----------

# TASK 8C: Display accuracy on HOLDOUT set of the optimized decision tree.


# COMMAND ----------

# MAGIC %md
# MAGIC Does the optimized decision tree perform better then the one with default parameters?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional Advanced TASK: Tuning Random Forest
# MAGIC When you are tuning a more complex model, it is a good practice to search available literature on which hyperparameters should be tuned. Below I have predefined some. You can play around with the grid, for example expand or narrow it. Keep in mind that as our feature set is extremely limited, its hard for hyperparameter tuning to arrive to something meaningful.

# COMMAND ----------

# OPTIONAL TASK 9
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#Define a pipeline
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

#Searching strategy, providing grid
tuning_rf = GridSearchCV(rf_pipeline, param_grid_rf)

#Train
tuning_rf.fit(X_train, y_train)

#Cross-validated score (more robust than holdout set most likely)
print(tuning_rf.best_score_)
print(tuning_rf.best_params_)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional Advanced TASK: Check Kaggle competitions and join one of them!  
# MAGIC https://www.kaggle.com/
