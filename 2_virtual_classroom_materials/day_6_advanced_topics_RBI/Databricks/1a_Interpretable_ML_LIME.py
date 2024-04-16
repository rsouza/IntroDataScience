# Databricks notebook source
# MAGIC %md
# MAGIC # Explaining your Machine Learning model using [LIME](https://ema.drwhy.ai/LIME.html)  
# MAGIC [Github](https://github.com/marcotcr/lime)  
# MAGIC
# MAGIC
# MAGIC ### [Why should you trust your model?](https://towardsdatascience.com/decrypting-your-machine-learning-model-using-lime-5adc035109b5)
# MAGIC
# MAGIC Shapley values are most suitable for models with a small or moderate number of explanatory variables. For models with a very large number of explanatory variables, sparse explanations with a small number of variables offer a useful alternative. The most popular example of such sparse explainers is the Local Interpretable Model-agnostic Explanations (LIME) method and its modifications.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 - [Intuition behind LIME](https://www.kaggle.com/code/prashant111/explain-your-model-predictions-with-lime/notebook)   
# MAGIC
# MAGIC ![](https://miro.medium.com/max/1165/1*k-rxjnvUDTwk8Jfg6IYBkQ.png)
# MAGIC
# MAGIC The intuition behind LIME is very simple. First, forget the training data and imagine we have only the black box model where we supply the input data. The black box model generate the predictions for the model. We can enquire the box as many times as we like. Our objective is to understand why the machine learning model made a certain prediction.
# MAGIC
# MAGIC Now, LIME comes into play. LIME tests what happens to the predictions when we provide variations in the data which is being fed into the machine learning model.
# MAGIC
# MAGIC LIME generates a new dataset consisting of permuted samples and the corresponding predictions of the black box model. On this new dataset LIME then trains an interpretable model. It is weighted by the proximity of the sampled instances to the instance of interest. The learned model should be a good approximation of the machine learning model predictions locally, but it does not have to be a good global approximation. This kind of accuracy is also called **local fidelity**. There is no dependency on the type of original model for LIME to provide explanations (model agnostic).
# MAGIC
# MAGIC ![https://towardsdatascience.com/decrypting-your-machine-learning-model-using-lime-5adc035109b5](https://miro.medium.com/max/720/1*vE3PUuhG6RRgK1J9oxg0nA.webp)
# MAGIC
# MAGIC What does LIME offer for model interpretability?
# MAGIC 1. A consistent model agnostic explainer – LIME.  
# MAGIC 2. A method to select a representative set with explanations – SP-LIME – to make sure the model behaves consistently while replicating human logic. This representative set would provide an intuitive global understanding of the model.  

# COMMAND ----------

!pip install -U -q lime
!pip install -U lightgbm

# COMMAND ----------

import lime
import lime.lime_tabular

import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 - Data Preprocessing and fitting a LightGBM model
# MAGIC
# MAGIC We are going to use the Titanic data set. At this point you should be fairly familiar with it. Most of the preprocessing should not contain any new material and you can focus on the sections using LIME. 

# COMMAND ----------

# reading the titanic data
df_titanic = pd.read_csv("../../../Data/data_titanic/train.csv")

df_titanic.head()

# COMMAND ----------

cols = df_titanic.columns 
colours = ['darkblue', 'red'] 
sns.heatmap(df_titanic[cols].isnull(), cmap=sns.color_palette(colours))

# COMMAND ----------

pct_list = []
for col in df_titanic.columns:
    pct_missing = np.mean(df_titanic[col].isnull())
    if round(pct_missing*100) >0:
        pct_list.append([col, round(pct_missing*100)])
    print('{} - {}%'.format(col, round(pct_missing*100)))

# COMMAND ----------

# MAGIC %md
# MAGIC The feature “Cabin” is missing 77% of the data. So we are going to remove that feature. 
# MAGIC
# MAGIC Age, however, is missing 20% of the data. Age should be an important variable in this application since it must have affected the probability of survival (e.g. older people or children might have been given the priority). Usually, we would just fill the missing values with the mean of the other’s people’s age. However, in this specific dataset, people were from different classes so it’s not a good idea to treat all of them as one group. The dataset has a feature “Name” the name has the title of the people (e.g. “Mr”, “Miss”…etc). That title should be a great indication of the age. Also, we should keep in mind that at that time of the incidence (in 1912) the socioeconomic status affected the people’s title regardless on age (e.g. younger people who are rich could get titles that usual poor people at the same age wouldn’t). So we are going to group people by their title and Pclass and then we will assign the mean of the age of each group to the missing age in each group.

# COMMAND ----------

# extracting the title from the name:
Title = []
for name in  df_titanic.Name:
    Title.append(name.split(",")[1].split(".")[0])
    
df_titanic["Title"] = Title

# COMMAND ----------

#grouping people with pclass and title
df_titanic.groupby(["Pclass", 'Title'])['Age'].agg(['mean']).round(0)

# adding the mean of the age of each group to the missing values
df_titanic["Age"] = df_titanic.groupby(["Title", "Pclass"])["Age"].transform(lambda x: x.fillna(x.mean()))

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we can also delete the unneeded features like the name (after extracting the title from it), the ticket ID, the passenger ID.

# COMMAND ----------

df_titanic.drop(columns = ["Name", "PassengerId", "Ticket", "Cabin", "Title"], inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC As a final step, we will encode the categorical features into numerical:

# COMMAND ----------

df_titanic.Sex = pd.Categorical(df_titanic.Sex)
df_titanic.Embarked = pd.Categorical(df_titanic.Embarked)
df_titanic["Sex"] = df_titanic.Sex.cat.codes
df_titanic["Embarked"] = df_titanic.Embarked.cat.codes

# COMMAND ----------

# MAGIC %md
# MAGIC **Using train test split to create validation set**

# COMMAND ----------

df_titanic.head()

# COMMAND ----------

feat = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]

X_train, X_test, y_train, y_test = train_test_split(
    df_titanic[feat], df_titanic[["Survived"]], test_size=0.3
)

# COMMAND ----------

# specify your configurations as a dict
lgb_params = {
    "task": "train",
    "data_sample_strategy": "goss",
    "objective": "binary",
    "metric": "binary_logloss",
    "metric": {"l2", "auc"},
    "num_leaves": 5,
    "learning_rate": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "verbose": 0,
    "num_iteration": 100,
    "num_threads": 7,
    "max_depth": 12,
    "alpha": 0.5,
}


# def lgb_model(X_train,y_train,X_test,y_test,lgb_params):
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)


# training the lightgbm model
model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=20,
    valid_sets=lgb_eval,
    callbacks=[
        lgb.early_stopping(stopping_rounds=5),
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3 - Model agnostic explainer (_LIME_).
# MAGIC **LIME requires class probabilities in case of classification example.**   
# MAGIC LightGBM directly returns probability for class 1 by default, so we will use it as a model here for simplicity.  

# COMMAND ----------

model.feature_name()

# COMMAND ----------

def prob(data):
    return np.array(list(zip(1-model.predict(data),model.predict(data))))
    
explainer = lime.lime_tabular.LimeTabularExplainer(df_titanic[model.feature_name()].astype(int).values,
                                                   mode='classification',
                                                   training_labels=df_titanic['Survived'],
                                                   feature_names=model.feature_name())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 - Asking for explanation for LIME model
# MAGIC
# MAGIC There are three parts to the explanation :
# MAGIC
# MAGIC + Left most section displays prediction probabilities.
# MAGIC + The middle section returns the 5 most important features. For the binary classification task, it would be in 2 colors. orange/blue. Attributes in orange support class 1 and those in blue support class 0. `Sex_le` ≤0 supports class 1. 
# MAGIC + Float point numbers on the horizontal bars represent the relative importance of these features.
# MAGIC + The color-coding is consistent across sections. It contains the actual values of the top 5 variables.

# COMMAND ----------

i = 1
exp = explainer.explain_instance(df_titanic.loc[i,feat].astype(int).values, prob, num_features=5)
exp.show_in_notebook(show_table=True)

# COMMAND ----------

i = 6
exp = explainer.explain_instance(df_titanic.loc[i,feat].astype(int).values, prob, num_features=5)
exp.show_in_notebook(show_table=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 - Submodular pick (*SP-LIME*) for explaining models
# MAGIC
# MAGIC LIME aims to attribute a model’s prediction to human-understandable features. In order to do this, we need to run the explanation model on a diverse but representative set of instances to return a nonredundant explanation set that is a global representation of the model.  
# MAGIC
# MAGIC **Note:** Running `SubmodularPick` can take some time, so you might want to run the cells below and return at a later point. 

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

from lime import submodular_pick

# Remember to convert the dataframe to matrix values
# SP-LIME returns explanations on a sample set to provide a non redundant global decision boundary of original model
sp_obj = submodular_pick.SubmodularPick(explainer, 
                                        df_titanic[model.feature_name()].values, 
                                        prob, 
                                        num_features=3,
                                        num_exps_desired=5)

for exp in sp_obj.sp_explanations:
    exp.show_in_notebook()
