# Databricks notebook source
# MAGIC %md
# MAGIC # Text classification
# MAGIC In this notebook, we're going to experiment with a few "traditional" approaches to text classification. These approaches pre-date the deep learning revolution in Natural Language Processing, but are often quick and effective ways of training a text classifier.

# COMMAND ----------

import os
import re
from zipfile import ZipFile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data
# MAGIC
# MAGIC We will be analyzing a dataset comprising nearly 21,000 titles and abstracts of research papers.
# MAGIC Our objective is to determine the topics of each article based on this data.
# MAGIC It is important to note that articles may be associated with multiple topics.
# MAGIC We will discuss the implications of this aspect in more detail later on.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading data files
# MAGIC
# MAGIC We begin by extracting the data from a zip files which contains the dataset in form of a csv file.

# COMMAND ----------

with ZipFile(os.path.join("../../../Data", "topics", 'train.csv.zip'), 'r') as myzip:
    with myzip.open('train.csv') as myfile:
        train_df = pd.read_csv(myfile)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's have a look at the structure of the data.

# COMMAND ----------

train_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC The last 6 columns encode the topics of the articles.

# COMMAND ----------

categories = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology", "Quantitative Finance"]

# COMMAND ----------

# MAGIC %md
# MAGIC Lets now check for NULL values and the data types of columns.  
# MAGIC (Sometimes columns which contain float or integer values are assigned the data type object. In that case we need to change the data type.)  
# MAGIC The [`info()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html) method conveniently gives us all of these informations plus the shape of the data frame and the memory usage.

# COMMAND ----------

train_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Preprocessing
# MAGIC The first step in the development of any NLP model is text preprocessing.
# MAGIC This means we're going to transform our texts from word sequences to feature vectors.
# MAGIC These feature vectors each contain the values of' a large number of features.  
# MAGIC
# MAGIC In this experiment, we're going to work with so-called **"bag-of-word"** approaches.
# MAGIC Bag-of-word methods treat every text as an unordered collection of words (or optionally, _ngrams_),
# MAGIC and the raw feature vectors simply tell us how often each word (or ngram) occurs in a text.
# MAGIC In Scikit-learn, we can construct these raw feature vectors with
# MAGIC [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html),
# MAGIC which tokenizes a text and counts the number of times any given text contains every token in the corpus.
# MAGIC During this step we'll also discard so called stop words.
# MAGIC Stop words are words like *and*, *the*, *her*, which are presumed to be uninformative in representing the content of a text.
# MAGIC Always be aware that these words are removed, as there is no general solution for this task.
# MAGIC
# MAGIC However, these raw counts are not very informative yet.
# MAGIC This is because the raw feature vectors of most texts, even though stop words are removed, in the same language will still be very similar.
# MAGIC We are interested in words that occur often in one text, but not very often in the corpus as a whole.
# MAGIC Therefore we're going to weight all features by their
# MAGIC [**tf-idf score**](https://en.wikipedia.org/wiki/Tf%E2%80%93idf),
# MAGIC which counts the number of times every token appears in a text and divides it by (the logarithm of) the percentage of corpus documents that contain that token.
# MAGIC This weighting is performed by Scikit-learn's
# MAGIC [`TfidfTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).  
# MAGIC
# MAGIC To obtain the weighted feature vectors, we combine the
# MAGIC [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
# MAGIC and
# MAGIC [`TfidfTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
# MAGIC in a Pipeline, and fit this pipeline on the training data.
# MAGIC Conveniently Scikit-learn has this Pipeline already implemented as
# MAGIC [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
# MAGIC We then transform both the training texts and the test texts to a collection of such weighted feature vectors.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleaning the text
# MAGIC
# MAGIC Before we vectorize our text we'll perform some manual cleaning.
# MAGIC Let's concatenate 'Title' and 'Abstract' and make it one big text.

# COMMAND ----------

train_df["text"] = train_df["TITLE"] + " " + train_df["ABSTRACT"]

# COMMAND ----------

# MAGIC %md
# MAGIC We drop the 'Title' and 'Abstract' columns as they are not needed anymore. 

# COMMAND ----------

train_df.drop(["TITLE","ABSTRACT"],axis=1,inplace=True)

# COMMAND ----------

train_df.head()

# COMMAND ----------

def clean_text(input_text):
    x = re.sub('[^\w]|_', ' ', input_text)  # only keep numbers and letters and spaces
    x = x.lower()
    x = re.sub(r'[^\x00-\x7f]',r'', x)  # remove non ascii texts
    x = [y for y in x.split(' ') if y] # remove empty words
    x = ['[number]' if y.isdigit() else y for y in x]
    cleaned_text =  ' '.join(x)
    return cleaned_text

# COMMAND ----------

train_df['cleaned_text'] = train_df['text'].apply(clean_text)

# COMMAND ----------

train_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Below we have a look at one of the cleaned texts.

# COMMAND ----------

train_df.cleaned_text[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Changing text into numericals using Tfidf technique
# MAGIC
# MAGIC Before we can apply the Tfidf technique we need to split the data into training and test sets.
# MAGIC Otherwise information from the test set would leak into the training data and any result would be spoilt.

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(train_df.loc[:,"cleaned_text"], train_df.loc[:,categories], test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Now that our text is cleaned we will apply Tfidf on the text data to convert it into a matrix of numericals.

# COMMAND ----------

tfidf = TfidfVectorizer(min_df=3, 
                        max_features=10000, 
                        strip_accents="unicode", 
                        analyzer="word",
                        token_pattern=r"\w{1,}",
                        ngram_range=(1,2),
                        use_idf=1,
                        smooth_idf=1,
                        sublinear_tf=1,
                        stop_words="english")

# COMMAND ----------

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# COMMAND ----------

X_train_tfidf.shape

# COMMAND ----------

X_test_tfidf.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Now we are done with preprocessing the text corpus.
# MAGIC We transformed each title + abstract of the articles to a numerical vecotor of length 10'000.
# MAGIC This representation enables us to train classifiers we alread know and use them for text classification.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multilabel (text) classification
# MAGIC
# MAGIC There are four main types of classification problems:
# MAGIC - **Binary**: The traget label has only two values and all observations belong to either one label or the other.
# MAGIC - **Multiclass**: The traget label has more than two values and all observations are assigned exactly one label.
# MAGIC - **Multilabel**: The target label has two or more values but any observation is assigned one or more labels.
# MAGIC - **Multitask**: The target is label each observation with multiple lables with non-binary properties.
# MAGIC
# MAGIC As mentioned earlier we are dealing with a multilabel classification task.
# MAGIC (Our target, `y_train` has multiple columns but each column is binary.)
# MAGIC Let's explore the data further and investigate the labels.

# COMMAND ----------

number_labels = y_train.sum(axis=1)
no_label_count = number_labels[number_labels < 1].count()

print("Number of articles in the training data = ",y_train.shape[0])
print("Total number of  training articles without label = ",no_label_count)
print("Total labels in training data = ", y_train.sum().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC We have more than 16,000 articles in the train data. 
# MAGIC All the articles are labeled under at least one topic.
# MAGIC There are some articles with more than one topic.
# MAGIC As our dataset contains articles with multiple tags, we are dealing with a **multi-label classification problem**.
# MAGIC
# MAGIC Let us plot a graph to look at the class distribution.

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's check each how many abstracts belongs to each category.

# COMMAND ----------

category_count = y_train.sum()
print(category_count)

# COMMAND ----------

plt.figure(figsize=(15,5))
plt.bar(category_count.index, category_count)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The plot indicates that “Quantitative Biology” and “Quantitative Finance” categories contain significantly fewer entries compared to other categories, highlighting an imbalance in the dataset. This imbalance poses a challenge in accurately predicting outcomes for the minority classes. Although multilabel classification tasks complicate the application of resampling techniques, it is important to be aware that [specialized methods](https://www.sciencedirect.com/science/article/abs/pii/S0950705115002737) do exist to tackle such imbalances. We will continue with the current dataset for this analysis, but these strategies should be considered for future refinement.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluating the model
# MAGIC Since we are dealing with a new kind of problem we also need to think about how we will measure the preformance of our models.
# MAGIC
# MAGIC To evaluate the performance we will be using three different metrics.
# MAGIC - [`accuracy_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html): 
# MAGIC In multilabel classification, this function computes subset accuracy.
# MAGIC The set of labels predicted for a sample must exactly match the corresponding set of labels in `y_true`.
# MAGIC - [`f1_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html):
# MAGIC By setting `average="macro"` the metric is calculating the mean of the f1 score for each class. This way each class is of equal importance and a dominant class has less influce.
# MAGIC - [`hamming_loss()`]():
# MAGIC The Hamming loss corresponds to the [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) between `y_true` and `y_pred`.
# MAGIC It ranges from 0 to 1, where a smaller value indicates better performance.
# MAGIC
# MAGIC Scikit-learn enables us to calculate a confusion matrix for each class by using [`multilabel_confusion_matrix()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html).
# MAGIC Below we have wrapped this function to automatically plot all of classes in a neat grid.

# COMMAND ----------

from math import ceil


def plot_confusion_matrices(y_true, y_pred, nrows=2, figsize=5):
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)

    number_matrices = y_true.shape[1]
    nrows = nrows
    ncols = ceil(number_matrices / nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize * ncols, figsize * nrows))

    for (i, confusion_matrix, title) in zip(
        np.arange(number_matrices), confusion_matrices, y_train.columns
    ):
        sns.heatmap(
            confusion_matrix,
            cmap="Blues",
            ax=axs[i // ncols, i % ncols],
            cbar=False,
            fmt="g",
            annot=True,
        )
        axs[i // ncols, i % ncols].set_title(title)
        axs[i // ncols, i % ncols].set_xlabel("Predicted")
        axs[i // ncols, i % ncols].set_ylabel("Actual")
    
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Now that all the components are in place we can finally begin to fit a model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Naive Approach
# MAGIC
# MAGIC There are multiple ways of dealing with such a classification task.
# MAGIC First we will showcase the most straight forward one.
# MAGIC For each of the classes we will train an independet binary classifier.
# MAGIC
# MAGIC The types of classifiers are explained in the next section.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Classifiers
# MAGIC We're going to experiment with three classic text classification models: Naive Bayes, Support Vector Machines and Logistic Regression. 
# MAGIC
# MAGIC [Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) are extremely simple classifiers that assume all features are independent of each other. They just learn how frequent all classes are and how frequently each feature occurs in a class. To classify a new text, they simply multiply the probabilities for every feature \\(x_i\\) given each class \\(C\\) and pick the class that gives the highest probability: 
# MAGIC
# MAGIC $$ \hat y = argmax_k \, [ \, p(C_k) \prod_{i=1}^n p(x_i \mid C_k)\, ]  $$
# MAGIC
# MAGIC Naive Bayes Classifiers are very quick to train, but usually fall behind in terms of performance.
# MAGIC
# MAGIC [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) are much more advanced than Naive Bayes classifiers. They try to find the hyperplane in the feature space that best separates the data from the different classes. They do so by picking the hyperplane that maximizes the distance to the nearest data point on each side. When the classes are not linearly separable, SVMs map the data into a higher-dimensional space where a linear separation can hopefully be found. SVMs often achieve very good performance in text classification tasks.
# MAGIC
# MAGIC [Logistic Regression models](https://en.wikipedia.org/wiki/Logistic_regression), finally, model the log-odds \\(l\\), or \\(\log[p\,/\,(1-p)]\\), of a class as a linear model and estimate the parameters \\(\beta\\) of the model during training: 
# MAGIC
# MAGIC \\(l = \beta_0 + \sum_{i=1}^n \beta_i x_i\\)
# MAGIC
# MAGIC Like SVMs, they often achieve great performance in text classification.

# COMMAND ----------

plt.ioff();

# COMMAND ----------

classifiers = {
    "nb_mo": MultiOutputClassifier(MultinomialNB()),
    "svm_mo": MultiOutputClassifier(LinearSVC()),
    "lr_mo": MultiOutputClassifier(LogisticRegression())
}

print("Training MultiOutput Naive Bayes classifier...")
classifiers["nb_mo"].fit(X_train_tfidf, y_train)

print("Training MultiOutput SVM classifier...")
classifiers["svm_mo"].fit(X_train_tfidf, y_train)

print("Training MultiOutput Logistic Regressor...")
classifiers["lr_mo"].fit(X_train_tfidf, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC We'll have a look at another approach and have a look at the results later on.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classifier chain approach
# MAGIC Often the labels are correlated.
# MAGIC With the previous naive approach we have dismissed such information entierly.
# MAGIC A classifier chain fits a classifier for each label sequentially and uses the prediction of all the previous labels as input as well to leverage correlation between labels.
# MAGIC
# MAGIC Let's fit the models and compare the results.

# COMMAND ----------

classifiers["nb_chain"] = ClassifierChain(MultinomialNB())
classifiers["svm_chain"] = ClassifierChain(LinearSVC())
classifiers["lr_chain"] = ClassifierChain(LogisticRegression())

print("Training chain Naive Bayes classifier...")
classifiers["nb_chain"].fit(X_train_tfidf, y_train)

print("Training chain SVM classifier...")
classifiers["svm_chain"].fit(X_train_tfidf, y_train)

print("Training chain Logistic Regressor...")
classifiers["lr_chain"].fit(X_train_tfidf, y_train)

# COMMAND ----------

plt.close("all")
plt.ion();

# COMMAND ----------

train_predictions = {type: classifiers[type].predict(X_train_tfidf) for type in classifiers}
test_predictions = {type: classifiers[type].predict(X_test_tfidf) for type in classifiers}

metrics = pd.DataFrame(
    {
        "Multilabel": ["MultiOutput"] * 3 + ["ClassifierChain"] * 3,
        "Classifier": ["Naive Bayes", "SVM", "Logistic Regression"] * 2,
        "Train Accuracy": [accuracy_score(y_train, train_predictions[type]) for type in train_predictions],
        "Train (Macro) F1 score": [f1_score(y_train, train_predictions[type], average="macro") for type in train_predictions],
        "Train Hamming Loss": [hamming_loss(y_train, train_predictions[type]) for type in train_predictions],
        "Test Accuracy": [accuracy_score(y_test, test_predictions[type]) for type in test_predictions],
        "Test (Macro) F1 score": [f1_score(y_test, test_predictions[type], average="macro") for type in test_predictions],
        "Test Hamming Loss": [hamming_loss(y_test, test_predictions[type]) for type in test_predictions]
    }
)

# metrics.set_index(["Multilabel", "Classifier"], inplace=True)

display(metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC - If we check out the results above we can see that the SVM classifiers perform well on the training data however they tend to overfit, since the results are not as good on the test set.
# MAGIC - On the other hand the Logistic Regression models have similar performance on the test set.
# MAGIC - For all classifiers it holds true that the chain approach performs better than the naive approach.
# MAGIC
# MAGIC Let's check out the confusion matrices for the chain Logistic Regression classifier.

# COMMAND ----------

plot_confusion_matrices(y_test, test_predictions["lr_chain"])

# COMMAND ----------

# MAGIC %md
# MAGIC The low number of true positives highlights the problem of the imbalanced data set.
# MAGIC There is not enough data to properly learn how to identify "Quantitative Biology" and "Quantitative Finance" articles.
# MAGIC
# MAGIC For all the results above keep in mind that the hyperparameters of the models are not tuned.
# MAGIC Feel free to experiment with the hyperparameters to see if you can improve the results.

# COMMAND ----------

# MAGIC %md
# MAGIC ## References
# MAGIC
# MAGIC * https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/  
# MAGIC * https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff  
# MAGIC * https://www.thepythoncode.com/article/text-classification-using-tensorflow-2-and-keras-in-python   
# MAGIC * https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles/code
