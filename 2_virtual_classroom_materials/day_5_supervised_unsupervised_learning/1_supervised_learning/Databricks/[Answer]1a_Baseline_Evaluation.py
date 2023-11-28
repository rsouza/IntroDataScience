# Databricks notebook source
# MAGIC %md
# MAGIC # Supervised Learning Workflow

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Baseline Model & Model Evaluation

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

train = pd.read_csv("../../../../Data/data_titanic/train.csv")
train.Pclass = train.Pclass.astype(float) # to avoid DataConversionWarning

# COMMAND ----------

train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Brief Exploration

# COMMAND ----------

# Categorical features
train.describe(include = object)

# COMMAND ----------

# Numerical features
train.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's work only with the following features for simplicity:   
# MAGIC
# MAGIC **Categorical**   
# MAGIC - Sex
# MAGIC - Embarked
# MAGIC
# MAGIC **Numerical**  
# MAGIC - Survived: *our target feature* (0 = No, 1 = Yes)
# MAGIC - Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
# MAGIC - Age: Age in years
# MAGIC  
# MAGIC More detailed info: https://www.kaggle.com/c/titanic

# COMMAND ----------

# Let's keep only the desired columns
train = train[['Sex','Embarked','Pclass', 'Age','Survived']]
train.shape

# COMMAND ----------

# Check for missing values
train.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC For simplicity, we drop any row containing missing values. 

# COMMAND ----------

train = train.dropna(axis=0)
train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC With our current knowledge, we can try to individually implement various transformers from Scikit Learn. Let's not forget to create a holdout set!

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(train[['Pclass', 'Age', 'Sex', 'Embarked']],
                                                    train['Survived'], 
                                                    test_size=0.2, 
                                                    random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical Features
# MAGIC The only numerical features we have are 'Pclass' and 'Age'.  
# MAGIC Let's scale these two features using `MinMaxScaler()`.

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
# MAGIC The categorical features we have are 'Sex' and 'Embarked'.   
# MAGIC We can simply one-hot encode these using `OneHotEncoder()`.

# COMMAND ----------

encoder = preprocessing.OneHotEncoder(sparse=False)
encoder.fit(X_train[['Sex', 'Embarked']])
X_train_transformed_categorical = encoder.transform(X_train[['Sex', 'Embarked']])
X_test_transformed_categorical = encoder.transform(X_test[['Sex', 'Embarked']])

print(X_train_transformed_categorical.shape)
print(X_test_transformed_categorical.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC It's time for our first exercise! 
# MAGIC Before, let's concatenate the transformed numerical and categorical features into a single dataframe.

# COMMAND ----------

X_train_transformed = np.concatenate((X_train_transformed_numerical, X_train_transformed_categorical), axis = 1)
X_test_transformed = np.concatenate((X_test_transformed_numerical, X_test_transformed_categorical), axis = 1)

print(X_train_transformed.shape)
print(X_test_transformed.shape)

# COMMAND ----------

# TASK 1A: Fit DummyClassifier to the transformed training set.  
# Then, let the model predict for train (X_train_transformed) and holdout set (X_test_transformed).
# Store the prediction as y_pred_TRAIN_DUMMY (training set) and as y_pred_HOLDOUT_DUMMY (holdout set).

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_transformed, y_train)

y_pred_TRAIN_DUMMY = dummy_clf.predict(X_train_transformed)
y_pred_HOLDOUT_DUMMY = dummy_clf.predict(X_test_transformed)

# COMMAND ----------

# OPTIONAL TASK 1B: Think about a simple heuristic that can be used as a baseline. 
# One possibility is to use gender and for example predict that every men or every woman has survived.
# You can store the result as y_pred_TRAIN_HEURISTIC and as y_pred_HOLDOUT_HEURISTIC.

y_pred_TRAIN_HEURISTIC =   np.array([1 if idx==0 else 0 for idx in X_train_transformed[:,3]])
y_pred_HOLDOUT_HEURISTIC = np.array([1 if idx==0 else 0 for idx in X_test_transformed[:,3]])

# COMMAND ----------

# MAGIC %md
# MAGIC Great! We have our first prediction! It is time to evaluate how good our model is using the [*sklearn.metrics* module.](   
# MAGIC https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)

# COMMAND ----------

#TASK 2A: Display ACCURACY on TRAIN set.
print(metrics.accuracy_score(y_train, y_pred_TRAIN_DUMMY))
print(metrics.accuracy_score(y_train,y_pred_TRAIN_HEURISTIC))  #Optional Task 1
print()

#TASK 2B: Display ACCURACY on HOLDOUT set.
print(metrics.accuracy_score(y_test, y_pred_HOLDOUT_DUMMY))
print(metrics.accuracy_score(y_test, y_pred_HOLDOUT_HEURISTIC))  #Optional Task 1

#OPTIONAL TASK 2C: Can you think of a better measure than accuracy based on the domain problem? If yes, use it the same way.

# COMMAND ----------

# MAGIC %md
# MAGIC Great! Now we would also like to see the confusion matrix as it is always a good idea to visually confirm the quality of our predictions.

# COMMAND ----------

#TASK 3: Display a CONFUSION MATRIX on HOLDOUT set. Hint: do not use plot_confusion_matrix but confusion_matrix only.
metrics.confusion_matrix(y_test, y_pred_HOLDOUT_DUMMY)

# COMMAND ----------

metrics.confusion_matrix(y_test, y_pred_HOLDOUT_HEURISTIC)
