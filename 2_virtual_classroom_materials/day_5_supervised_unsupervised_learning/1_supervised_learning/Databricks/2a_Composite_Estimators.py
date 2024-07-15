# Databricks notebook source
!pip install --upgrade scikit-learn
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Supervised Learning Workflow
# MAGIC Let's continue with our previous example and see how we can use composite estimators for our problem.

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train = pd.read_csv("../../../../Data/data_titanic/train.csv")
train.Pclass = train.Pclass.astype(float) # to avoid DataConversionWarning
train = train[['Sex','Embarked','Pclass', 'Age','Survived']]

# COMMAND ----------

# MAGIC %md
# MAGIC **Note**   
# MAGIC If you later want to experiment with the composite transformers, comment out this cell and include also missing value imputation.

# COMMAND ----------

train = train.dropna(axis=0)
train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Composite Estimators
# MAGIC Let's nicely wrap our feature engineering and model fitting into a nice composite estimator. We will be very simplistic and only use two steps. 
# MAGIC They will not nest into each other at once.

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(train[['Pclass', 'Age', 'Sex', 'Embarked']],
                                                    train['Survived'], 
                                                    test_size=0.2, 
                                                    random_state=42)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering wrapped into ColumnTransformer
# MAGIC The two feature transformations can be easily wrapped up into a single
# MAGIC [`ColumnTransformer()`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
# MAGIC object.
# MAGIC This will ensure that our Feature Engineering is a bit **more robust and nicely encapsulated**.
# MAGIC Section 6.1.4 [here](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)
# MAGIC showcases the exact application that we intend to create.

# COMMAND ----------

# TASK 1: Wrap MinMaxScaler and OneHotEncoder into a single ColumnTransformer. 
# The transformers should be applied to the respective numerical or categorical columns only.
# Store the resulting composite as feature_engineering
# Hint: Use the argument remainder='passthrough'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predictive Model Wrapped into Pipeline
# MAGIC Let's now wrap the feature engineering and the model into a single Pipeline Composite estimator. Here is some pseudocode for this:
# MAGIC ``` 
# MAGIC entire_pipeline = feature_engineering -> model  
# MAGIC ``` 
# MAGIC
# MAGIC Both components are already available. From the step above we can directly reuse the object `feature_engineering`.
# MAGIC As model, we just call a new
# MAGIC [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html),
# MAGIC just as we did before.

# COMMAND ----------

# TASK 2: Wrap the feature engineering and the predictive model (dummy) into a single Pipeline composite estimator. 
# Store the result as entire_pipeline.

# COMMAND ----------

# TASK 3: Uncomment the line and try to train the pipeline.
# Notice that we are using untransformed data again (X_train) as the pipeline contains all necessary transformers.

# entire_pipeline.fit(X = X_train, y = y_train)

# COMMAND ----------

# Predict for training data
y_pred_TRAIN_DUMMY = entire_pipeline.predict(X_train)

# Predict for holdout data
y_pred_HOLDOUT_DUMMY = entire_pipeline.predict(X_test)

# Results should be the same as before
print(metrics.accuracy_score(y_train, y_pred_TRAIN_DUMMY))

# Display accuracy on holdout set.
print(metrics.accuracy_score(y_test, y_pred_HOLDOUT_DUMMY))

# COMMAND ----------

# MAGIC %md
# MAGIC **OPTIONAL TASK**   
# MAGIC The notebook <a href="$./2b_Example_Pipelines">``2b_Example_Pipelines``</a> was made to exemplify some examples of more complex pipelines. Feel free to scroll through it and learn what the process of preparing a complex composite looks like. You can then come back here and try to implement various components. For example, if I would not drop rows with missing values at the beginning of this notebook, constructing a composite would get a bit trickier. 
