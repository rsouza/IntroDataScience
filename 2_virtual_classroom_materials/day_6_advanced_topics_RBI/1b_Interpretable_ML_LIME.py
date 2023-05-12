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

# COMMAND ----------

import lime
import lime.lime_tabular

import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 - Data Preprocessing

# COMMAND ----------

# reading the titanic data
df_titanic = pd.read_csv("../../Module_B/Day2/data/Titanic/kaggle_titanic_train.csv")


# data preparation
df_titanic.fillna(0,inplace=True)
df_titanic.head()

# COMMAND ----------

le = LabelEncoder()

feat = ['PassengerId', 'Pclass_le', 'Sex_le','SibSp_le', 'Parch','Fare']

# label encoding textual data
df_titanic['Pclass_le'] = le.fit_transform(df_titanic['Pclass'])
df_titanic['SibSp_le'] = le.fit_transform(df_titanic['SibSp'])
df_titanic['Sex_le'] = le.fit_transform(df_titanic['Sex'])

# COMMAND ----------

# MAGIC %md
# MAGIC **Using train test split to create validation set**

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(df_titanic[feat],df_titanic[['Survived']],test_size=0.3)

# COMMAND ----------

# specify your configurations as a dict
lgb_params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'binary',
    'metric':'binary_logloss',
    'metric': {'l2', 'auc'},
    'num_leaves': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': 0,
    'num_iteration':100,
    'num_threads':7,
    'max_depth':12,
    'min_data_in_leaf':100,
    'alpha':0.5}


# def lgb_model(X_train,y_train,X_test,y_test,lgb_params):
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)


# training the lightgbm model
model = lgb.train(lgb_params,
                  lgb_train,
                  num_boost_round=20,
                  valid_sets=lgb_eval,
                  early_stopping_rounds=5
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

#[exp.as_pyplot_figure(label=1) for exp in sp_obj.sp_explanations]
[exp.show_in_notebook() for exp in sp_obj.sp_explanations]
