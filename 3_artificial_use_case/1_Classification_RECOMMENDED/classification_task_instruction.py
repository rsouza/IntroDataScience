# Databricks notebook source
# MAGIC %md
# MAGIC # <center >Classification Task Instruction </center>

# COMMAND ----------

# MAGIC %md
# MAGIC In this task, you will get a sense of using Python to solve a classification problem.
# MAGIC The dataset that we will be working with is from a Portuguese banking institution about its direct marketing campaigns. <br>
# MAGIC ````
# MAGIC  Goal: to predict, if a client will subscribe a term deposit (denoted in **variable y**) or not.
# MAGIC ````
# MAGIC 
# MAGIC Remember that this is an artifical use case which is supposed to serve a **Contributor level** purposes. Thereafter, you should practice skills such as **Data Preprocessing, Data Visualisation, Preparing Data for ML** and only after you receive feedback on your initial work it is recommended to try **to fit some baseline Machine Learning Model**.
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dataset Information
# MAGIC 
# MAGIC Inside **Bank_Dataset** folder is your datasets, there are two files:
# MAGIC - `short` contains only a 10% of observations, is a subset of the full file.
# MAGIC - `full` contains all observations

# COMMAND ----------

# MAGIC %md
# MAGIC The variables explanation below is taken **directly** from the dataset source. 
# MAGIC 
# MAGIC ### Bank client data:
# MAGIC - `age` (numeric)
# MAGIC - `job` type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# MAGIC - `marital` marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# MAGIC - `education` (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# MAGIC - `default` has credit in default? (categorical: 'no','yes','unknown')
# MAGIC - `housing` has housing loan? (categorical: 'no','yes','unknown')
# MAGIC - `loan` has personal loan? (categorical: 'no','yes','unknown')
# MAGIC 
# MAGIC ### Related with the last contact of the current campaign:
# MAGIC - `contact` contact communication type (categorical: 'cellular','telephone')
# MAGIC - `month` last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# MAGIC - `day_of_week` last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# MAGIC - `duration` last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# MAGIC 
# MAGIC ### Other attributes :
# MAGIC - `campaign` number of contacts performed during this campaign and for this client (numeric, includes last contact)
# MAGIC - `pdays` number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# MAGIC - `previous` number of contacts performed before this campaign and for this client (numeric)
# MAGIC - `poutcome` outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# MAGIC     
# MAGIC ### Social and economic context attributes
# MAGIC - `emp.var.rate` employment variation rate - quarterly indicator (numeric)
# MAGIC - `cons.price.idx` consumer price index - monthly indicator (numeric)
# MAGIC - `cons.conf.idx` consumer confidence index - monthly indicator (numeric)
# MAGIC - `euribor3m` euribor 3 month rate - daily indicator (numeric)
# MAGIC - `nr.employed` number of employees - quarterly indicator (numeric)
# MAGIC 
# MAGIC Output variable (desired target):
# MAGIC - `y` has the client subscribed a term deposit? (binary: 'yes','no')
# MAGIC 
# MAGIC ## 2. Importing The Dataset
# MAGIC To import a dataset, you will need to specify the file where is dataset is located, and it should starts with the path as follows:
# MAGIC ````python
# MAGIC '/home/jovyan/work/....csv', error_bad_lines=False,  sep = ';'
# MAGIC ````
# MAGIC This is specific to our Workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Task Instruction
# MAGIC The steps below are served as a **guide** to solve this problem. They are by no means a must or the only way to solve this partcular dataset. Feel free to use what you have learned in the previous classrooms and to be creative. Try to find out your own approach to this problem.
# MAGIC 
# MAGIC **Step 1: Data Loading & Preprocessing**
# MAGIC - load the dataset into your Python Notebook
# MAGIC - convert the dataset to the desired format that you want to work with (dataframe, numpy.array, list, etc.)
# MAGIC - explore the dataset
# MAGIC - observe the variables carefully, and try to understand each variable and its business value
# MAGIC - don't forget the special treatment to null values
# MAGIC 
# MAGIC 
# MAGIC **Step 2: Data Visualisation & Exploration**
# MAGIC - employ various visualisation skills that you acquired inside and outside the classroom
# MAGIC - with the visualisation tools, understand what is happening in the dataset
# MAGIC 
# MAGIC 
# MAGIC **Step 3: Data modelling**
# MAGIC - separate variables & labels
# MAGIC - split dataset into training & testing dataset
# MAGIC - pick one data modelling approach respectively the Python modelling package that you would like to use
# MAGIC - fit the training dataset to the model and train the model
# MAGIC - output the model 
# MAGIC - make prediction on testing dataset
# MAGIC 
# MAGIC 
# MAGIC **OPTIONAL: Step 4: Fine tune the model or use more advanced modelling approaches**
# MAGIC - map the prediction of the testing dataset against real numbers from your dataset and compare the result
# MAGIC - make adjustments on your model for a better result (but make sure don't overfit the model)
# MAGIC 
# MAGIC 
# MAGIC **Step 5: Result extration & interpretation**
# MAGIC - make your conclusions and interpretation on the model and final results
# MAGIC - evaluate the performance of your model and algorithm using different KPIs 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Additional Resources
# MAGIC - you can take a look at one of the use cases that AA tribe had done in the past and get a sense of how the modelling approach feels like: general description of the modelling approach: <https://wiki.rbinternational.corp/confluence/display/AAT/Modelling+approach+RBBG>    
# MAGIC     - random forest approach: <https://wiki.rbinternational.corp/confluence/display/AAT/Random+Forest+Modelling+Approach>
# MAGIC     - gradient boosting approach:  <https://wiki.rbinternational.corp/confluence/display/AAT/Gradient+Boosting+Modelling+Approach>
# MAGIC - some of the modelling methods that you can go over:
# MAGIC     - random forest : https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
# MAGIC         - for random forest, please first understand [decision trees](https://en.wikipedia.org/wiki/Decision_tree#:~:text=A%20decision%20tree%20is%20a,only%20contains%20conditional%20control%20statements.) completely
# MAGIC     - gradient boosting: https://www.youtube.com/watch?v=3CC4N4z3GJc
# MAGIC     - support vector machine: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
# MAGIC 
# MAGIC **Packages that might be useful for you:**
# MAGIC - pandas: https://pandas.pydata.org/pandas-docs/stable/reference/index.html
# MAGIC - numpy: https://numpy.org/doc/
# MAGIC - scikit-learn: https://scikit-learn.org/stable/
# MAGIC - sklearn.linear_model: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# MAGIC - sklearn.datasets: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
# MAGIC - sklearn.ensemble: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# MAGIC - sklearn.preprocessing: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# MAGIC - sklearn.dummy: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
# MAGIC - sklearn.metrics: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# MAGIC 
# MAGIC 
# MAGIC **Useful links:**
# MAGIC - Die Pipeline: https://wiki.rbinternational.corp/confluence/display/AAT/Die+Pipeline
# MAGIC - Importing data with read_csv: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# MAGIC - subsetting dataset in pandas: https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
# MAGIC - column transformers with mixed types: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
# MAGIC - feature scaling for machine learning: https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
# MAGIC - what is ont-hot encoding: https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/
# MAGIC - xgboost model with python: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# MAGIC - random forest classifier: https://www.datacamp.com/community/tutorials/random-forests-classifier-python

# COMMAND ----------

# MAGIC %md
# MAGIC **Dataset citation:**
# MAGIC [[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
