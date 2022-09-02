# Databricks notebook source
# MAGIC %md
# MAGIC # Clustering Notebook
# MAGIC 
# MAGIC In this task, we will work on some clustering problems. The dataset we will be working on is a 52-week of sales transaction report. The **ultimate goal** of this task is to find similar time series in the sales transaction data.
# MAGIC 
# MAGIC ## Dataset Information
# MAGIC We have only one csv file prepared for you. There are 53 attributes with 811 entries (or rows) in the file.
# MAGIC The first column is Product_Code, then we have the rest of 52 columns corresponding to 52 weeks of our sales transactions. The normalised values are also provided.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task: Clustering
# MAGIC The steps below are served as a guidance to solve this clustering problem. They are by no means a must or the only way to solve this partcular dataset. Feel free to use what you have learned in the previous program and to be creative. Try to find out your own approach to this problem.
# MAGIC 
# MAGIC 1. **Step 1: Data loading & preprocessing**
# MAGIC     - load the data into Python Notebook and convert it to the appropriate format (dataframe, numpy.array, list, etc.)
# MAGIC     - observe & explore the dataset
# MAGIC     - check for null values
# MAGIC 
# MAGIC 2. **Step 2: Data modelling**
# MAGIC     - standardize the data to normal distribution 
# MAGIC     - pick one data modelling approach respectively the Python modelling package that you would like to use
# MAGIC     - fit the training dataset to the model and train the model
# MAGIC     - output the model 
# MAGIC 
# MAGIC 4. **Step 3: Result extration & interpretation**
# MAGIC     - make your conclusions and interpretation on the model and final results
# MAGIC     - evaluate the performance of your model and algorithm using different KPIs 
# MAGIC    
# MAGIC **Packages that might be useful for you:**
# MAGIC - pandas
# MAGIC - numpy
# MAGIC - sklearn
# MAGIC - sklearn.cluster
# MAGIC - matplotlib
# MAGIC    
# MAGIC ### Additional Resources:
# MAGIC 
# MAGIC **Useful links:**
# MAGIC - k-means clustering: https://en.wikipedia.org/wiki/K-means_clustering
# MAGIC - hierarchical clustering: https://en.wikipedia.org/wiki/Hierarchical_clustering & https://towardsdatascience.com/understanding-the-concept-of-hierarchical-clustering-technique-c6e8243758ec
# MAGIC - the elbow method: https://en.wikipedia.org/wiki/Elbow_method_(clustering)#:~:text=In%20cluster%20analysis%2C%20the%20elbow,number%20of%20clusters%20to%20use.
# MAGIC 
# MAGIC     

# COMMAND ----------

# MAGIC %md
# MAGIC **Dataset citation:**
# MAGIC @inproceedings{tan2014time,
# MAGIC title={Time series clustering: A superior alternative for market basket analysis},
# MAGIC author={Tan, Swee Chuan and San Lau, Jess Pei},
# MAGIC booktitle={Proceedings of the First International Conference on Advanced Data and Information Engineering (DaEng-2013)},
# MAGIC pages={241--248},
# MAGIC year={2014},
# MAGIC organization={Springer, Singapore}
# MAGIC }
