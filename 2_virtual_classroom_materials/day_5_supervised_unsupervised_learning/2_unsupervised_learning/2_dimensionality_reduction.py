# Databricks notebook source
# MAGIC %md
# MAGIC # Dimensionality reduction

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
from sklearn import datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ## PCA
# MAGIC 
# MAGIC Principal component analysis is an unsupervised learning method that tries to detect the directions in which the vector formed data varies most

# COMMAND ----------

# MAGIC %md
# MAGIC import data of digits in black and white colors. 

# COMMAND ----------

digits = datasets.load_digits()
x_digits, y_digits = digits.data, digits.target

# COMMAND ----------

# MAGIC %md
# MAGIC Let's start by visualizing our data. Fetch the first 10 numbers. 

# COMMAND ----------

_, axes = plt.subplots(nrows=1, ncols=10, figsize=(16, 5))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: return the shape of x_digits

# COMMAND ----------

# Task
x_digits.shape

# COMMAND ----------

# MAGIC %md
# MAGIC There are 1797 numbers represented by 8 x 8 matrices with the color intensity for each pixel. These are flattened a vector of 64 numbers

# COMMAND ----------

# MAGIC %md
# MAGIC Our data has 64 dimensions, but we are going to reduce it to only 2 and see that, even with just 2 dimensions, we can clearly see that digits separate into clusters.
# MAGIC 
# MAGIC __TO DO__: 
# MAGIC - import PCA from scikit-learns decomposition module
# MAGIC - fit_transform PCA() model with n_components = 2 on x_digits and assign the result to the variable x_reduced

# COMMAND ----------

# Task

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_reduced = pca.fit_transform(x_digits)

# COMMAND ----------

# MAGIC %md
# MAGIC if we plot the reduced 2D feature space we see that similar numbers are close to each other

# COMMAND ----------

plt.figure(figsize=(12,8))
plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y_digits, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('MNIST. PCA projection');

# COMMAND ----------

# MAGIC %md
# MAGIC In practice, we would choose the number of principal components such that we can explain e.g. 90% of the initial data dispersion (via the explained_variance_ratio_). 
# MAGIC 
# MAGIC __TO DO__: 
# MAGIC - return the explained variance ratio with PCA()'s attribute explained_variance_ratio_.  
# MAGIC Each value returns the percentage of variance explained by each of the selected components. 
# MAGIC - How much variance did we explain with the first two components?
# MAGIC 
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

# COMMAND ----------

# Task

print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))

# COMMAND ----------

# MAGIC %md
# MAGIC The length of the returned array is equal to n_components. If n_components is not set, then all components are stored and the sum of the ratios is equal to 1.0.

# COMMAND ----------

# MAGIC %md
# MAGIC In order to get 90% explained variance we would have to retain 20 principal components. Run the code below to see.

# COMMAND ----------

pca = PCA().fit(x_digits)

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, x_digits.shape[1])
plt.yticks(np.arange(0, 1.1, 0.1))
min_variance = 0.9
min_components = np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > min_variance))
print(min_components)
plt.axvline(min_components, c='b')
plt.axhline(min_variance, c='r')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC __scaling__
# MAGIC 
# MAGIC In the MNIST dataset all the features have the same scale so the PCA algorithm is not impacted by different scales. 
# MAGIC 
# MAGIC While many algorithms (such as SVM, K-nearest neighbors, and logistic regression) require features to be normalized, intuitively we can think of Principle Component Analysis (PCA) as being a prime example of when normalization is important. In PCA we are interested in the components that maximize the variance. If one component (e.g. human height) varies less than another (e.g. weight) because of their respective scales (meters vs. kilos), PCA might determine that the direction of maximal variance more closely corresponds with the ‘weight’ axis, if those features are not scaled. As a change in height of one meter can be considered much more important than the change in weight of one kilogram, this is clearly incorrect.

# COMMAND ----------

from sklearn.datasets import load_wine

wine = load_wine()
x_wine = wine.data
y_wine = wine.target
cols_wine = wine.feature_names
df_wine = pd.DataFrame(x_wine, columns = cols_wine)

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: use a describe() method on df_wine to see standard deviations, quartiles and other descriptive statistics of the columns. 

# COMMAND ----------

# Task

df_wine.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC notice that the scales of the features are different e.g. proline is in magnitude of hundreds, magnesium in tens, other are even smaller

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: 
# MAGIC - fit_tranform the PCA with 2 components on the x_wine
# MAGIC - return the reduced data set with 2 features and assign to the variable _x_reduced_
# MAGIC - print the components of the pca with components_ attribute

# COMMAND ----------

# Task

pca_unscaled = PCA(n_components=2)
x_reduced = pca_unscaled.fit_transform(x_wine)
pca_unscaled.components_

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: 
# MAGIC - use StandardScaler() to scale (fit_transform) the x_wine and store it to x_wine_scaled variable
# MAGIC - fit_transform the PCA with 2 components on the scaled data
# MAGIC - return the reduced data set with 2 features and assign it to the variable x_reduced_scaled
# MAGIC - print the components of the pca with components_ attribute

# COMMAND ----------

# Task

x_wine_scaled = StandardScaler().fit_transform(x_wine)
pca_scaled = PCA(n_components=2)
x_reduced_scaled = pca_scaled.fit_transform(x_wine_scaled)
print(pca_scaled.components_)

# COMMAND ----------

# MAGIC %md
# MAGIC observe above how the last value of the components in the unscaled case was by cca 2 magnitudes larger than the other values of the components. In the scaled case the components have similar weights. You can see the results of the two PCAs in the chart below.

# COMMAND ----------

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (15,6))

ax1.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y_wine, 
            edgecolor='none', alpha=0.7, s=40)
ax1.set_title('wine PCA projection');

ax2.scatter(x_reduced_scaled[:, 0], x_reduced_scaled[:, 1], c=y_wine, 
            edgecolor='none', alpha=0.7, s=40)
ax2.set_title(f'wine PCA projection scaled');

# COMMAND ----------

# MAGIC %md
# MAGIC Scaling helped to visualize the similar results close to each other. As a consequence scaling improves the performance of classifiers after dimensionality reduction.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# COMMAND ----------

# MAGIC %md
# MAGIC # Optional part 

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: use a train_test_split to split the x_wine, y_wine data into x_train, x_test, y_train, y_test

# COMMAND ----------

# Task

x_train, x_test, y_train, y_test = train_test_split(x_wine, y_wine, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Here we will make use of make_pipeline which constructs a pipeline from the given estimators. First we will apply PCA with 2 components to reduce dimensionality of the training data x_train and then apply RandomForestClassifier() with default parameters. We initialize PCA and RF in the pipeline and fit it to the training data.
# MAGIC 
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html

# COMMAND ----------

clf_unscaled = make_pipeline(PCA(2), RandomForestClassifier()) 
clf_unscaled.fit(x_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will predict the classes with the pipeline on a test dataset x_test and evaluate by comparing the predictions to true values from y_test

# COMMAND ----------

pred_test_unscaled = clf_unscaled.predict(x_test)
accuracy_score(y_test, pred_test_unscaled)

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__:  repeat the same steps as in the previous two cells 
# MAGIC - make pipeline but now include additional estimator _StandardScaler()_ at the beginning of the pipeline. Assign the pipeline to the variable clf_scaled. In the example above you had 2 estimators (PCA(2) and RandomForestClassifier()). Now you will have 3 estimators. 
# MAGIC - fit the pipeline to the training data x_train and y_train
# MAGIC - use the predict() method of clf_scaled on the test data x_test and store the predictions to a variable pred_test_scaled
# MAGIC - print the accuracy of the pred_test_scaled compared to y_test

# COMMAND ----------

# Task

clf_scaled = make_pipeline(StandardScaler(), PCA(2), RandomForestClassifier())
clf_scaled.fit(x_train, y_train)
pred_test_scaled = clf_scaled.predict(x_test)
accuracy_score(y_test, pred_test_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that scaling improved the accuracy of the model significantly

# COMMAND ----------

# MAGIC %md
# MAGIC ## TSNE
# MAGIC 
# MAGIC TSNE is another dimensionality reduction method that works well for visualizing high-dimensional data. With t-sne you can get different results with different initializations. If the number of features is high t-sne can get quite slow. Here we select 2000 random samples from the mnist digits dataset and still t-sne takes much more time to complete than PCA. 

# COMMAND ----------

from sklearn.manifold import TSNE
from datetime import datetime as dt

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: use TSNE with n_components = 2 and fit_transform the random 2000 examples from mnist data set that is stored in x_digits_flat_selected

# COMMAND ----------

# Task

t1 = dt.now()
x_embedded = TSNE(n_components=2).fit_transform(x_digits)
plt.figure(figsize=(10,7))
plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y_digits, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('MNIST. TSNE projection');
print(f'Timerun: {dt.now()-t1}')

# COMMAND ----------

# MAGIC %md
# MAGIC With t-SNE, the picture looks better since PCA has a linear constraint while t-SNE does not. Details of the algorithm are for further reading. 
# MAGIC 
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Autoencoders

# COMMAND ----------

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf

current_dir = os.getcwd()
current_dir = os.path.join(current_dir, "data")
current_dir

# COMMAND ----------

(x_digits, y_digits), (_, _) = tf.keras.datasets.mnist.load_data(path=current_dir+'/mnist.npz')

# if you have problems with RAM memory run the 3 lines below that will select just a subset of the data

idxs_ = random.sample(range(x_digits.shape[0]), 10000)
x_digits = x_digits[idxs_]
y_digits = y_digits[idxs_]

# COMMAND ----------

# MAGIC %md
# MAGIC Let's start by visualizing our data. Fetch the first 10 numbers. 

# COMMAND ----------

plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_digits[i,:].reshape([28,28]), cmap='gray');

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: return the shape of x_digits

# COMMAND ----------

# Task

x_digits.shape

# COMMAND ----------

# MAGIC %md
# MAGIC There are 60.000 numbers represented by 28 x 28 matrices with the color intensity for each pixel. We need to flatten every matrix into a vector of 28x28 numbers. We will use .reshape() function that gives a new shape to an array without changing its data
# MAGIC 
# MAGIC https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

# COMMAND ----------

x_digits_flat = x_digits.reshape(x_digits.shape[0], x_digits.shape[1]*x_digits.shape[2]) # reshaping matrices into vectors
x_digits_flat.shape

# COMMAND ----------

# MAGIC %md
# MAGIC compile the encoder decoder architecture stacked sequentially (autoencoder) using keras. if it is not installed you will have to install keras either with pip or conda. just execute the code and read comments. learning keras is out of scope of this lecture

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Reshape
from tensorflow.keras.optimizers import SGD, Adam
import seaborn as sns

X_train = x_digits/255.0 # digits matrices are scaled to values between 0 and 1
 
### Encoder
encoder = Sequential()
encoder.add(Flatten(input_shape=[28,28]))   # flatten the input matrices into vectors of 28x28
encoder.add(Dense(400,activation="relu"))   # add a dense layer with 400 neurons and relu activation
encoder.add(Dense(200,activation="relu"))
encoder.add(Dense(100,activation="relu"))
encoder.add(Dense(50,activation="relu"))
encoder.add(Dense(2,activation="relu"))     # add a dense layer with 2 neurons and relu activation
 
### Decoder
decoder = Sequential()
decoder.add(Dense(50,input_shape=[2],activation='sigmoid'))  # decoder will start with an input of dimension 2 from an encoder
decoder.add(Dense(100,activation='sigmoid'))
decoder.add(Dense(200,activation='sigmoid'))
decoder.add(Dense(400,activation='sigmoid'))
decoder.add(Dense(28 * 28, activation="sigmoid"))
decoder.add(Reshape([28, 28]))           # reshape the output from a flat vector to matrix of 28x28 
 
### Autoencoder
autoencoder = Sequential([encoder,decoder]) # stack encoder and decoder sequentially
autoencoder.compile(loss="mse", optimizer='Adam') # compile the model

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: fit the compiled autoencoder model as you are used from scikit learn. as x use X_train and as y use X_train again, specify number of epochs with parameter epochs=5

# COMMAND ----------

# Task

autoencoder.fit(X_train,X_train,epochs=5)

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: use a predict() method of the encoder on X_train and assign the result to x_reduced

# COMMAND ----------

# Task

x_reduced = encoder.predict(X_train) 

# COMMAND ----------

plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y_digits, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('Encoder PCA projection');

# COMMAND ----------

# MAGIC %md
# MAGIC After training with 5 epochs the result does not look very promising. However, if you let the model learn for 50 epochs you should get the result as shown on image below. 

# COMMAND ----------

autoencoder.fit(X_train,X_train,epochs=50)

# COMMAND ----------

x_reduced = encoder.predict(X_train)

# COMMAND ----------

plt.figure(figsize=(12,12))
plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y_digits, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('Encoder PCA projection');

# COMMAND ----------

# MAGIC %md
# MAGIC Some material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)