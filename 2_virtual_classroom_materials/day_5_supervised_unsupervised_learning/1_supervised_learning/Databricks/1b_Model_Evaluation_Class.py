# Databricks notebook source
# MAGIC %md
# MAGIC # Model Evaluation: Classification

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluating the performance of classification models is a critical aspect of building effective predictive models. In this notebook we are going to explore various metrics for binary classification tasks:
# MAGIC
# MAGIC * **Accuracy**
# MAGIC
# MAGIC * **F1 score**
# MAGIC
# MAGIC * **ROC AUC**
# MAGIC
# MAGIC * **PR AUC**
# MAGIC
# MAGIC
# MAGIC Which of these metrics is better? When and how should we use them? What are they good for?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Accuracy
# MAGIC
# MAGIC Accuracy measures the proportion of correctly predicted instances among the total instances in a dataset. In other words, accuracy tells you how well the model's predictions match the actual outcomes.
# MAGIC
# MAGIC **Accuracy = (Number of Correct Predictions)/(Total Number of Predictions)**
# MAGIC
# MAGIC
# MAGIC When does it make sense to use it?
# MAGIC * You should not use accuracy on imbalanced datasets. In situations where the classes are imbalanced, meaning one class has significantly more instances than the others, accuracy can be misleading. A model might achieve high accuracy by simply predicting the majority class all the time.
# MAGIC * You can use it when every class is equally important to you and errors are equally costly. In some applications, the cost of making a particular type of error might be much higher than the others. For instance, in medical diagnosis, a false negative (saying a person is healthy when they are not) can be more critical than a false positive. Accuracy doesn't take into account the severity of different types of errors.

# COMMAND ----------

# MAGIC %md
# MAGIC ## F1 Score
# MAGIC
# MAGIC The F1 score is the **harmonic mean of precision and recall**. 
# MAGIC
# MAGIC Precision is the ratio of true positive predictions to the total predicted positives.
# MAGIC
# MAGIC **Precision = tp/(tp+fp)**
# MAGIC
# MAGIC Recall is the ratio of true positive predictions to the total actual positives.
# MAGIC
# MAGIC **Recall = tp/(tp+fn)**
# MAGIC
# MAGIC The formula for calculatng the F1 score is:
# MAGIC
# MAGIC **F1 = 2 x (Precision x Recall)/(Precision + Recall)**
# MAGIC
# MAGIC When does it make sense to use it?
# MAGIC * When the positive class is more important for you.
# MAGIC * In scenarios where the costs of different types of errors (false positives and false negatives) are not equal, the F1 score provides a balanced assessment of the trade-off between precision and recall. For instance, in medical diagnoses, a false negative (missing a disease) could be more severe than a false positive (wrongly diagnosing a healthy patient).
# MAGIC * In fraud detection or anomaly detection, the class of interest (fraudulent cases or anomalies) is often the minority class. The F1 score is better suited to evaluate the model's performance in correctly identifying these rare occurrences.
# MAGIC * In tasks like sentiment analysis, where the classes might not be perfectly balanced, the F1 score helps assess the model's performance when dealing with different levels of sentiment expressions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ROC AUC Score
# MAGIC
# MAGIC
# MAGIC The Receiver Operating Characteristic (ROC) curve and the Area Under the ROC Curve (ROC AUC) are popular evaluation metrics used in binary classification tasks. The ROC AUC score is particularly useful for assessing a model's ability to discriminate between positive and negative classes across different probability thresholds. 
# MAGIC
# MAGIC The ROC AUC score measures the area under the ROC curve. The ROC curve is a graphical representation that illustrates the trade-off between:
# MAGIC
# MAGIC * true positive rate (TPR, recall)
# MAGIC
# MAGIC * false positive rate (FPR)
# MAGIC
# MAGIC **FPR = FP/(FP + TN)**
# MAGIC
# MAGIC as the classification threshold changes.
# MAGIC
# MAGIC AUC is the probability that the model ranks a random positive example more highly than a random negative example. 
# MAGIC
# MAGIC *More about ROC AUC: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc*
# MAGIC
# MAGIC When does it make sense to use it?
# MAGIC * You should not use it with heavily imbalanced dataset. False positive rate for highly imbalanced datasets is pulled down due to a large number of true negatives.
# MAGIC * You should use it when you ultimately care about ranking predictions and not necessarily about outputting well-calibrated probabilities

# COMMAND ----------

# MAGIC %md
# MAGIC ## PR AUC Score
# MAGIC
# MAGIC Precision-recall curve combines precision (PPV) and Recall (TPR) in a single visualization. The higher on y-axis your curve is the better your model performance. The higher the recall, the lower the precision. The PR AUC score measures the area under this curve. It quantifies the balance between precision and recall across various thresholds and provides a single value that ranges between 0 and 1. 
# MAGIC
# MAGIC When does it make sense to use it?
# MAGIC * The PR AUC score is especially relevant for imbalanced datasets, where the positive class (minority class) is significantly outnumbered by the negative class. 
# MAGIC * When the positive class is of particular interest, such as detecting rare diseases, fraud cases, anomalies, or relevant search results, the PR AUC score becomes essential. It emphasizes the model's ability to accurately identify positive instances while keeping false positives low.
# MAGIC * When you want to choose the threshold that fits the business problem.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Fraud Detection

# COMMAND ----------

# MAGIC %md
# MAGIC We will work with a dataset https://www.kaggle.com/competitions/ieee-fraud-detection/data to detect fraudulent transactions. The dataset was transformed (https://github.com/neptune-ai/blog-binary-classification-metrics), so that it has 43 features, 66000 observations and the fraction of the positive class is 0.09.

# COMMAND ----------

# Import all necessary libraries
import pandas as pd
import lightgbm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve

# COMMAND ----------

train = pd.read_csv('../../../../Data/fraud_train.csv')
test = pd.read_csv('../../../../Data/fraud_test.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Brief exploration

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at our dataset.

# COMMAND ----------

train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will check if our dataset is balanced.

# COMMAND ----------

class_counts = train['isFraud'].value_counts()
class_counts

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that our data is **imbalanced**: fraudulent transactions (class = 1) account for one tenth of non-fraudulent transactions (class = 0).

# COMMAND ----------

train.shape

# COMMAND ----------

test.shape

# COMMAND ----------

feature_names = [col for col in train.columns if col not in ['isFraud']]

# Split dataset
X_train, y_train = train[feature_names], train['isFraud']
X_test, y_test = test[feature_names], test['isFraud']

# COMMAND ----------

# Create a list of different hyperparameters
parameters = [
  {
    'learning_rate': 0.1,
    'n_estimators': 10
  },
  {
    'learning_rate': 0.1,
    'n_estimators': 100
  },
  {
    'learning_rate': 0.1,
    'n_estimators': 300
  },
  {
    'learning_rate': 0.1,
    'n_estimators': 600
  },
  {
    'learning_rate': 0.1,
    'n_estimators': 1500
  },
  {
    'learning_rate': 0.05,
    'n_estimators': 1500
  },
  {
    'learning_rate': 0.05,
    'n_estimators': 3000
  }
]

# COMMAND ----------

results = []
id = 0

# Create models for different hyperparameters
for config in parameters:
  model = lightgbm.LGBMClassifier(random_state = 42, learning_rate = config['learning_rate'], n_estimators = config['n_estimators'])
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:, 1]

  accuracy = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_prob)
  pr_auc = average_precision_score(y_test, y_prob)
  
  # Store results
  result = {
      'ID': id,
      'Learning rate': config['learning_rate'],
      'N_estimators': config['n_estimators'],
      'Accuracy': accuracy,
      'F1 Score': f1,
      'ROC AUC': roc_auc,
      'PR AUC': pr_auc,
  }
  results.append(result)
  id += 1


# COMMAND ----------

# MAGIC %md
# MAGIC Let us take a look at the results and make conclusions.

# COMMAND ----------

for result in results:
  print(result)

# COMMAND ----------

# Sort experiments by accuracy
sorted(results, key=lambda x: x.get("Accuracy"))

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that our models always have high accuracy score (>90%). The worst model has the accuracy of 0.93. But remember that we have an imbalanced dataset. It means that even if all transactions will be classified as non-fraudulent, we will get an accuracy of 0.9. **You should always take an imbalance into consideration when looking at accuracy!**

# COMMAND ----------

sorted(results, key=lambda x: x.get("F1 Score"))

# COMMAND ----------

# MAGIC %md
# MAGIC With the imbalance we have, even the worst model has very high accuracy and the improvements as we go to the end of the table are not as clear on accuracy as they are on F1 score. Hence, between accuracy and F1 Score one should choose **F1 Score** for this dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC As a next step we will compare ROC AUC and PR AUC.

# COMMAND ----------

sorted(results, key=lambda x: x.get("ROC AUC"))

# COMMAND ----------

sorted(results, key=lambda x: x.get("PR AUC"))

# COMMAND ----------

# MAGIC %md
# MAGIC ROC AUC and PR AUC both assess the performance of classification models based on prediction scores rather than fixed class assignments. However, they differ in the metrics they focus on. ROC AUC examines the True Positive Rate (TPR) and False Positive Rate (FPR), whereas PR AUC considers the Positive Predictive Value (PPV) and TPR.
# MAGIC
# MAGIC If your primary concern is the positive class, PR AUC is often a superior choice. It is more sensitive to improvements in the positive class, making it particularly valuable in scenarios with highly imbalanced datasets. For instance, in cases like fraud detection, where the positive class (i.e., instances of fraud) is rare compared to the negative class, PR AUC provides a more informative evaluation of model performance.
# MAGIC
# MAGIC In our case we can see, that although ROC AUC and PR AUC rank models in the same way, the improvements calculated in **PR AUC** are larger and clearer. We get from 0.69 to 0.87 when at the same time ROC AUC goes from 0.92 to 0.96.

# COMMAND ----------

# MAGIC %md
# MAGIC Now we should decide between F1 Score and PR AUC.
# MAGIC
# MAGIC One significant distinction between the F1 score and ROC AUC is that the F1 score operates on predicted classes, while ROC AUC relies on predicted scores. Consequently, when using the F1 score, you must select a threshold for class assignment, a decision that can substantially impact model performance.
# MAGIC
# MAGIC If your objective is to rank predictions, without the need for well-calibrated probabilities, and your dataset maintains a reasonable balance between classes, then ROC AUC is a good choice.
# MAGIC
# MAGIC However, in scenarios characterized by a heavily imbalanced dataset, or when your primary concern centers on the positive class, considering the F1 score or Precision-Recall curve with PR AUC is advisable. Additionally, using F1 may be advantageous because this metric is more straightforward to interpret and convey to business stakeholders.

# COMMAND ----------

# MAGIC %md
# MAGIC *This notebook is based on the article: https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc.*
