# Databricks notebook source
# MAGIC %md
# MAGIC # Clustering

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## K-means

# COMMAND ----------

# MAGIC %md
# MAGIC The scikit-learn library has an implementation of the k-means algorithm. Let’s apply it to a set of randomly generated blobs, whose labels we throw away. 
# MAGIC 
# MAGIC make_blobs() function generates a data set for clustering. 
# MAGIC 
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
# MAGIC 
# MAGIC __TO DO__: Find out how many instances with how many features were generated. hint: shape

# COMMAND ----------

# Task
from sklearn.datasets import make_blobs
X,y = make_blobs(random_state=42) 
##

# COMMAND ----------

# MAGIC %md
# MAGIC Plot the points X with a scatter plot

# COMMAND ----------

plt.scatter(X[:,0],X[:,1]);

# COMMAND ----------

# MAGIC %md
# MAGIC In this toy example you can guess the number of clusters. Let’s see if the k-means algorithm can recover these clusters. 
# MAGIC 
# MAGIC __TO DO__: Import KMeans from sklearn.cluster and create the instance of the k-means model by giving it the number of clusters 3 as a hyperparameter. Fit the model to your dataset X. Notice that we do not feed labels y into the model. Help yourself with the examples from the documentation
# MAGIC 
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# COMMAND ----------

# Task


# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: assign centroids to the variable named _centroids_ and print it. Use KMeans model's attribute cluster_centers_. 
# MAGIC 
# MAGIC - The centroids are important because they are what enables KMeans to assign new, previously unseen points to the existing clusters

# COMMAND ----------

# Task



# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: assign predicted class labels to the variable _labels_ and print it. Use KMeans model's attribute labels_

# COMMAND ----------

# Task



# COMMAND ----------

# MAGIC %md
# MAGIC Plot the points X with a scatter plot with colors equal to labels. The centroids are plotted with a red dot.

# COMMAND ----------

plt.scatter(X[:,0],X[:,1], c=labels);
plt.scatter(centroids[:,0], centroids[:,1], s=100, color="red"); # Show the centres

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: return KMeans' performance measure inertia  with an attribute inertia_
# MAGIC 
# MAGIC - inertia = Sum of squared distances of samples to their closest cluster center. The lower the inertia the better. 

# COMMAND ----------

# Task



# COMMAND ----------

# MAGIC %md
# MAGIC Select the number of clusters where inertia does not decrease significantly anymore = Elbow rule.

# COMMAND ----------

inertia = []
for k in range(1, 15):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    inertia.append(km.inertia_)
    
plt.plot(range(1, 15), inertia, marker='s')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('$J(C_k)$');

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: Use the .predict() method of model to predict the cluster labels of new_points. Assign them to variable named new_labels. Notice that KMeans can assign previously unseen points to the clusters it has already found!

# COMMAND ----------

# Task

new_points = np.array([[-4, 5], [-6, -2.5], [4, -2.5], [0, 10]])
###
print(new_labels)

plt.scatter(X[:,0],X[:,1], c=labels)
plt.scatter(new_points[:,0], new_points[:,1], c=new_labels, marker = 'X', s = 300);

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scaling, numerical variables
# MAGIC 
# MAGIC Read the data set below. 

# COMMAND ----------

# https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
customers = pd.read_csv('data/Mall_Customers.csv')
customers.set_index('CustomerID', inplace = True)
customers['Annual Income (k$)'] = customers['Annual Income (k$)']*1000
customers.columns = ['gender', 'age', 'annual_income_$', 'spending_score']
display(customers.head(5))
print(f'\nShape of customers data set: {customers.shape}')

# COMMAND ----------

# MAGIC %md
# MAGIC Annual income has a very different scale than other two numerical features. 
# MAGIC 
# MAGIC For distance based methods scaling helps if the features have very different scales. We will scale here all numerical features with StandardScaler(). It standardizes features by removing the mean and scales to unit variance
# MAGIC 
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_cols = customers.select_dtypes(exclude='object').columns # get numerical columns
customers[num_cols] = scaler.fit_transform(customers[num_cols]) # apply scaler to numerical columns
customers.head()

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: fit the KMeans() model to the customers dataset. You will get an error. Don't worry, this one is on purpose. Proceed with the next cell after the error.

# COMMAND ----------

# Task


# COMMAND ----------

# MAGIC %md
# MAGIC What is the problem here? We have to transform non-numerical columns to numerical ones as it is not possible to calculate a distance between non-numerical features. 

# COMMAND ----------

# use get_dummies from pandas to encode values from non-numerical columnns to numerical
# https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
customers = pd.get_dummies(customers, drop_first=True)

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: Fit the KMeans model again to the customers data

# COMMAND ----------

# Task


# COMMAND ----------

# MAGIC %md
# MAGIC ## DBSCAN  
# MAGIC #### This part is voluntary. Do it if you have managed to finish the first part before the time limit  
# MAGIC 
# MAGIC Density-Based Spatial Clustering of Applications with Noise. The algorithm finds core samples of high density and expands clusters from them. It is good for data which contain clusters of similar density.

# COMMAND ----------

from sklearn.cluster import DBSCAN
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC Import dataset and plot it. Clearly there should be two clusters

# COMMAND ----------

x_moons, y_moons = datasets.make_moons(n_samples=1000, noise=0.05,random_state=42)
plt.scatter(x_moons[:, 0], x_moons[:, 1], c=y_moons);

# COMMAND ----------

# MAGIC %md
# MAGIC KMeans fails to find appropriate clusters as it searches for convex shapes. See below.

# COMMAND ----------

kmeans_labels = KMeans(2).fit(x_moons).labels_
plt.scatter(x_moons[:, 0], x_moons[:, 1], c=kmeans_labels);

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: 
# MAGIC - Initiate DBSCAN instance with eps=0.05 and min_samples=5 and assign it to _dbscan_ variable. 
# MAGIC - Fit the model to x_moons.  

# COMMAND ----------

# Task

###
###

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: 
# MAGIC - Return labels of dbscan with an attribute labels_ and assign the labels to _dbscan_labels_ variable

# COMMAND ----------

# Task


# COMMAND ----------

# MAGIC %md
# MAGIC Notice that there are labels == -1. These denote outliers

# COMMAND ----------

unique, counts = np.unique(dbscan_labels, return_counts=True)
display(pd.DataFrame(np.asarray((unique, counts)).T, columns = ['labels', 'frequency']))

# COMMAND ----------

# MAGIC %md
# MAGIC The clusters and outliers on a plot

# COMMAND ----------

plt.scatter(x_moons[:, 0], x_moons[:, 1], c=dbscan_labels, alpha=0.7);

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: indices of the core instances are available in the core_sample_indices_ attribute of DBSCAN. Assign them to comps_idx variable

# COMMAND ----------

# Task


# COMMAND ----------

# MAGIC %md
# MAGIC The core instances are stored in the components_ attribute. Below we create a mask for indices which are core instances

# COMMAND ----------

print(dbscan.components_)

comps_idx_boolean = np.array([(i in comps_idx) for i in range(len(x_moons))]) # this creates a boolean mask for core instances 

# COMMAND ----------

plt.scatter(x_moons[comps_idx_boolean, 0], x_moons[comps_idx_boolean, 1], c='r', alpha=0.3, label='core')
plt.scatter(x_moons[~comps_idx_boolean, 0], x_moons[~comps_idx_boolean, 1], c='b', alpha=0.6, linewidths = 0.1, label='outliers')
plt.legend()
plt.title('Core vs non-core instances');

# COMMAND ----------

# MAGIC %md
# MAGIC The DBSCAN clustering did not return expected results. Let's try different eps and min_samples
# MAGIC 
# MAGIC __TO DO__: fit DBSCAN to x_moons again but now with eps=0.2 and min_samples=5. Plot the resulting clusters

# COMMAND ----------

# Task

###
###
###

# COMMAND ----------

# MAGIC %md
# MAGIC notice that DBSCAN class does not have .predict() method although it has a fit_predict() method. It cannot predict a cluster for a new instance

# COMMAND ----------

dbscan.predict(x_moons)

# COMMAND ----------

# MAGIC %md
# MAGIC Some material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)
