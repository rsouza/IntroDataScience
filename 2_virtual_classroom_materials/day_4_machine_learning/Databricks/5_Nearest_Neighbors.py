# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # k-Nearest Neighbors
# MAGIC
# MAGIC
# MAGIC ## 1. Overview
# MAGIC
# MAGIC
# MAGIC In this notebook, we are going to explore the k-Nearest Neighbors (kNN) algorithm, a simple yet powerful machine learning technique. The core principle of the kNN algorithm revolves around the concept of similarity. It assumes that data points that are close to each other in the feature space share similar characteristics or belong to the same class. 
# MAGIC When presented with a new data point, kNN identifies its k nearest neighbors from the training data based on a specified distance metric (e.g., Euclidean, Manhattan). These neighbors' classes or values are then used to determine the classification or regression output for the new data point.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Two critical parameters in kNN are:
# MAGIC - **k**: The number of nearest neighbors to consider. Choosing an appropriate value for k is essential, as it directly impacts the model's performance and generalization ability.
# MAGIC - **Distance Metric**: The method used to measure the similarity between data points. The choice of distance metric depends on the nature of the data and the problem domain.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. kNN for classification task
# MAGIC
# MAGIC In scikit-learn, implementing kNN for classification tasks is straightforward using the [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) class. It provides flexibility in choosing the value of k and the distance metric. 
# MAGIC
# MAGIC
# MAGIC Let's start with importing all necessary libraries:

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC Now we want to generate synthetic data for classification. We generate synthetic data using the [`make_classification`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) function. This function allows us to create a random n-class classification problem with specified features:

# COMMAND ----------

X, y = make_classification(n_samples=100, n_features=2, n_classes=2, 
                           n_clusters_per_class=1, n_redundant=0, random_state=30)

# Splitting data into training and testign sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# COMMAND ----------

# MAGIC %md
# MAGIC *Note: we should normalize the features when using a kNN classifier. To prevent data leakage, normalization should be done after splitting the dataset into the train and test sets.*

# COMMAND ----------

# MAGIC %md
# MAGIC Let's have a look at our generated data:

# COMMAND ----------

# Visualizing the generated data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Data')
plt.show()

# COMMAND ----------

# Visualizing the generated test data
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Basic kNN-classifier
# MAGIC
# MAGIC We will now implement the kNN classifier with default parameters (k = 5, metric = minkowski):

# COMMAND ----------

# Implementing basic kNN classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

# Making predictions on the test set
predictions = knn_classifier.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of kNN Classifier with 5 neighbors: {accuracy:.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC Since our test set size is 20 points and the accuracy is 90%, our model misclassified 2 data points.
# MAGIC
# MAGIC But how does the classification of a new observation actually work?
# MAGIC First the k nearest neighbors, according to the specified metric, are searched in the training data.
# MAGIC With the k nearest observations the algorithm performs majority voting.
# MAGIC If there is a tie, either because there are more than two classes or an even numbered k, we randomly pick one of the top classes.
# MAGIC Sometimes the voting is augmented by a weight function but more on that later.
# MAGIC
# MAGIC Let's visualize the decision boundary of the classifier. 
# MAGIC The code below creates a meshgrid of points covering the feature space, predicts the class labels for each point, and plots the decision boundary along with the data points.
# MAGIC

# COMMAND ----------

plt.figure(figsize=(10, 6))
h = 0.02 
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))

# Plotting decision boundary
Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.8)
plt.contour(xx, yy, Z, colors="black", linewidths=0.2)

# Plotting data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision boundary')
plt.show();

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Choosing parameters

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, using the default classifier is very easy. However, we know that kNN has two important parameters - the number of neighbors and the distance metric. What should we consider when choosing the **number of neighbors**? 
# MAGIC * A small value of K in k-Nearest Neighbors can be influenced by outliers, potentially leading to incorrect classifications.
# MAGIC * Conversely, a large K value might explore data too far from the point in consideration, reducing the accuracy of classifications.
# MAGIC * Small K values are computationally efficient, while large K values can become computationally expensive.
# MAGIC
# MAGIC Another important decision is chosing a **distance metric**. The most popular once are:
# MAGIC * [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance):
# MAGIC   * Suitable for datasets with continuous features.
# MAGIC   * Works well when the features have similar scales.
# MAGIC   * May not perform optimally if the data has outliers or is not linearly separable.
# MAGIC
# MAGIC * [Manhattan Distance](https://en.wikipedia.org/wiki/Taxicab_geometry):
# MAGIC   * Suitable for high-dimensional data or data with many categorical features.
# MAGIC   * Less sensitive to outliers compared to Euclidean distance.
# MAGIC   * Works well when the data lies on a grid or when the distances along different dimensions are not directly comparable.
# MAGIC
# MAGIC * [Minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance):
# MAGIC   * Adaptable to datasets with mixed feature types, accommodating both continuous and categorical variables.
# MAGIC   * Particularly useful when diverse feature scales or customized distance measurements are required.
# MAGIC
# MAGIC
# MAGIC In our case euclidean distance is the most suitable metric. 
# MAGIC
# MAGIC
# MAGIC What about the number of neighbors? Now it is your turn to experiment with different k!

# COMMAND ----------

#Task 1: implement the classifier with 3 neighbors and euclidean distance metric

knn_classifier3 = ...
knn_classifier3.fit(X_train, y_train)

# Making predictions on the test set
predictions = ...

# Calculating accuracy
accuracy = ...
print(f"Accuracy of kNN Classifier with 3 neighbors: {accuracy:.0%}")

#Task 2: implement the classifier with 7 neighbors and euclidean distance metric
knn_classifier7 = ...
knn_classifier7.fit(X_train, y_train)

# Making predictions on the test set
predictions = ...

# Calculating accuracy
accuracy = ...
print(f"Accuracy of kNN Classifier with 7 neighbors: {accuracy:.0%}")

# COMMAND ----------

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
k_list = [3, 5, 7]
# Create classifiers and plot decision boundaries
for i, classifier in enumerate([knn_classifier3, knn_classifier, knn_classifier7]):
    # Plot decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    axes[i].contour(xx, yy, Z, colors="black", linewidths=0.2)
    axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    axes[i].set_xlim(xx.min(), xx.max())
    axes[i].set_ylim(yy.min(), yy.max())
    axes[i].set_title(f'Decision Boundary (k = {k_list[i]})')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The visualizations above illustrate a key concept: as the parameter k increases, the decision boundary becomes progressively smoother. This is because a larger k value means that the classification of a new point is determined by considering a larger number of neighboring points, which inherently leads to more generalized decision boundaries.
# MAGIC
# MAGIC Another approach, which gives more reliable and robust result, is cross-validation. It serves a more comprehensive understanding of how different k values affect model performance across different subsets of the data. We'll explore the range of k-values from 1 to 30 to observe how accuracy fluctuates:

# COMMAND ----------

k_values = range(1, 31)
accuracy_scores = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    # The dataset will be divided into cv=5 equal-sized folds
    # with each fold serving as the testing set once and the remaining folds as the training set
    # the accuracy score of the classifier is computed on the corresponding testing set
    accuracy = cross_val_score(knn_classifier, X, y, cv=5)
    # The mean accuracy across the folds will be added to the list
    accuracy_scores.append(np.mean(accuracy))

sns.lineplot(x = k_values, y = accuracy_scores, marker = 'o')
plt.xlabel("k-values")
plt.ylabel("Accuracy Score")
plt.xticks(range(1, 31, 2))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The maximum accuracy score of 96% can be reached with 2 or 5 neighbors.

# COMMAND ----------

# Training the model with the optimal k-value

max_accuracy = np.max(accuracy_scores)
best_k = accuracy_scores.index(max_accuracy)+1

knn_classifier_uniform = KNeighborsClassifier(n_neighbors=best_k, metric = 'euclidean')
knn_classifier_uniform.fit(X_train, y_train)

predictions = knn_classifier_uniform.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the best kNN Classifier with {best_k} neighbors: {accuracy:.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC Another parameter of
# MAGIC [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
# MAGIC is `weights`. 
# MAGIC It affects which weight function is used in prediction and can take two options: 
# MAGIC
# MAGIC * uniform (by default), i.e. all points in each neighborhood are weighted equally
# MAGIC * distance, i.e. closer neighbors of a query point will have a greater influence than neighbors which are further away.
# MAGIC
# MAGIC Let's try using a classifier with the weight function based on distance and compare it to a classifier with uniform weights:

# COMMAND ----------

#Task 3: Train your model using the best k value we found with cross-validation and weights based on distances

knn_classifier_distance = ...
knn_classifier_distance.fit(X_train, y_train)

# Make predictions on the test set
predictions = knn_classifier_distance.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of kNN Classifier with {best_k} neighbors and weights based on distance: {accuracy:.0%}")

# COMMAND ----------

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
k_list = ['uniform', 'distance']
# Create classifiers and plot decision boundaries
for i, classifier in enumerate([knn_classifier_uniform, knn_classifier_distance]):
    # Plot decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    axes[i].contour(xx, yy, Z, colors="black", linewidths=0.2)
    axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    axes[i].set_xlim(xx.min(), xx.max())
    axes[i].set_ylim(yy.min(), yy.max())
    axes[i].set_title(f'Decision Boundary (weights = {k_list[i]})')

plt.show();

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that the weights parameter in kNN affects the decision boundary. With "uniform" weights, all nearest neighbors carry equal weight in decisions. Conversely, with "distance" weighting, closer neighbors hold more influence due to their inverse relationship with distance. Using weights based on distances may enhance model performance in some cases.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 [OPTIONAL] RadiusNeighborsClassifier
# MAGIC There is one more nearest neighbors classifier implemented in scikit-learn -
# MAGIC [`RadiusNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html).
# MAGIC While
# MAGIC [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
# MAGIC relies on the number of nearest neighbors k, 
# MAGIC [`RadiusNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html)
# MAGIC operates based on the number of neighbors within a fixed radius of each training point, where 'r' is a floating-point value specified by the user.
# MAGIC The parameter `outlier_label` determines how to deal with observations, where no training data is within the radius.

# COMMAND ----------

from sklearn.neighbors import RadiusNeighborsClassifier

radius = np.arange(0.5, 2, 0.1)
accuracy_scores = []

# Looking for the best radius using cross-validation
for r in radius:
    knn_classifier = RadiusNeighborsClassifier(radius=r, outlier_label="most_frequent")
    accuracy = cross_val_score(knn_classifier, X, y, cv=5)
    # Sometimes during cross-validation there may be no points in the given radius
    # To prevent the error, we need to clean our accuracy list from nan-values
    accuracy = [x for x in accuracy if not np.isnan(x)]
    accuracy_scores.append(np.mean(accuracy))

max_accuracy = np.max(accuracy_scores)
best_radius = radius[accuracy_scores.index(max_accuracy)]

knn_classifier = RadiusNeighborsClassifier(radius = best_radius, outlier_label="most_frequent")
knn_classifier.fit(X_train, y_train)

predictions = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Best accuracy of kNN Classifier with radius {best_radius:.2}: {accuracy:.0%}")

# Visualizing accuracy score with different radius-values
sns.lineplot(x = radius, y = accuracy_scores, marker = 'o')
plt.xlabel("Radius")
plt.ylabel("Accuracy Score")
plt.xticks(np.arange(0.5, 2, 0.1))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Compare the graph above with the graph showing how accuracy changes for different values of k.

# COMMAND ----------

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
k_list = ['k neighbors classifier', 'Radius neighbors classifier']
# Create classifiers and plot decision boundaries
for i, classifier in enumerate([knn_classifier_uniform, knn_classifier]):
    # Plot decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    axes[i].contour(xx, yy, Z, colors="black", linewidths=0.2)
    axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    axes[i].set_xlim(xx.min(), xx.max())
    axes[i].set_ylim(yy.min(), yy.max())
    axes[i].set_title(f'{k_list[i]}')

plt.show();

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. kNN for regression task

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's implement kNN regression. The algorithm calculates the distance between the input data point and all other data points in the training set. It then selects the k-nearest neighbors and averages their target values to predict the target value for the input data point.
# MAGIC
# MAGIC Hint: Check out the
# MAGIC [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
# MAGIC or the
# MAGIC [User Guide](https://scikit-learn.org/stable/modules/neighbors.html#regression)
# MAGIC for additional information.

# COMMAND ----------

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

# Generate synthetic dataset
np.random.seed(0)
X = np.random.rand(100, 1) * 10  
y = 2 * X.squeeze() + np.random.normal(0, 2, size=X.shape[0]) 

# Visualize the dataset
plt.scatter(X, y, color='blue')
plt.title('Generated Data')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# COMMAND ----------

# Task 4.1: Split the dataset into training and testing sets

...

# COMMAND ----------

# Task 4.2: Create and train the basic kNN-regressor

knn_regressor = ...

# COMMAND ----------

# Task 5: Make predictions on the test set and evaluate the model

predictions = ...

mse = ...
r2 = ...
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# COMMAND ----------

#Task 6: Perform cross-validation for each value of k

# Create a range of k values for parameter tuning
k_values = ...

# List to store mean squared errors for each value of k
mse_values = []

for k in k_values:
    knn_regressor = ...
    # Note that here we need to invert the score, since we are using the neg_mean_squared_error scoring function
    mse = -cross_val_score(knn_regressor, X, y, cv=5, scoring='neg_mean_squared_error')
    ...

# Visualize MSE
...

# COMMAND ----------

# Task 7: Create and train a model with an optimal k-value

# Find an optimal k from the array mse_values
smallest_mse = ...
best_k = ...

# Implement and fit the model
knn_regressor = ...

# Make predictions
...

# Evaluate the model
mse = ...
r2 = ...
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's have a look at the final model, we have learned.

# COMMAND ----------

# Visualize the data
plt.scatter(X, y, color='blue')
plt.title('Final Model')
plt.xlabel('X')
plt.ylabel('y')

# Add the model
X_vis = np.arange(0, 10, 0.01).reshape(-1,1)
y_vis = knn_regressor.predict(X_vis)
plt.plot(X_vis, y_vis, c="r")

plt.show()
