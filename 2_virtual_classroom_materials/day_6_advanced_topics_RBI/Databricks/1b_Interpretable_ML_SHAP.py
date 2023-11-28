# Databricks notebook source
# MAGIC %md
# MAGIC # Explaining Machine Learning models using [SHAP](https://ema.drwhy.ai/shapley.html)  
# MAGIC
# MAGIC SHAP is a great model interpretation tool. Even though it’s a sophisticated model, it’s intuitive to understand.  
# MAGIC SHAP’s goal is to provide a visualization of the effect of each feature on the outcome variable.   
# MAGIC
# MAGIC To do that
# MAGIC 1. **SHAP builds a model that uses all the features except the one of interest** and see **how the model would perform without that feature**. 
# MAGIC 2. Then, it would build the model again and do the prediction with the feature. 
# MAGIC 3. The effect of the feature would then be the difference between the two values. 
# MAGIC
# MAGIC The order at which features are passed to the model affects the output (especially in the tree-based models in which the model follows a schematic approach ordered by the features). So, SHAP computes all the possible permutation at which the different features can be passed to the model. This seems to have a huge computational cost but SHAP has optimized algorithms that make it faster for specific machine learning models.
# MAGIC
# MAGIC This notebook is partially based on this [blog post](https://towardsdatascience.com/using-model-interpretation-with-shap-to-understand-what-happened-in-the-titanic-1dd42ef41888).

# COMMAND ----------

!pip install -U -q shap

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 - Data Preprocessing
# MAGIC
# MAGIC We are going to use the Titanic data set. At this point you should be fairly familiar with it. Most of the preprocessing should not contain any new material and you can focus on the sections using SHAP. 

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
# MAGIC We will drop the “Survival” outcome variable from the data set.

# COMMAND ----------

target = df_titanic.Survived.values
df_titanic.drop(columns =["Survived"], inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 - Building a Linear Model
# MAGIC Finally, we are going to build the model. We will go with a simple logistic regression model since the goal here is to see how the features affect the outcome and not to obtain a high score in prediction.

# COMMAND ----------

# MAGIC %md
# MAGIC **Using train test split to create validation set**

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(df_titanic, target, test_size=0.3)

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, y_train)
LR.score(X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 - Using SHAP

# COMMAND ----------

import shap

# COMMAND ----------

explainer = shap.LinearExplainer(LR, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_size=[12,8])
shap.summary_plot(shap_values, X_train, plot_type="bar", plot_size=[12,8])

# COMMAND ----------

# MAGIC %md
# MAGIC + "Pclass" has a significant effect on the survival rate of the passengers. It’s the second most significant feature after “Sex”. 
# MAGIC + We see from the plot above that low values (blue) for "Pclass" which correspond to a class of 1 (richer people) have a positive effect on people’s survival while higher values (red), which correspond to the third class, have a negative effect on the survival rate. 
# MAGIC + We can also see that “sex” is the most important feature with an indication that being a “female” (blue) had a positive impact on the survival rate. 
# MAGIC + The feature “Age” also shows that lower values (blue) had a positive impact on survival.

# COMMAND ----------

# MAGIC %md
# MAGIC Let’s take a look at the variable "Fare" which is how much each person paid for their ticket. This variable should be a continuous description of people’s wealth:

# COMMAND ----------

shap.dependence_plot("Fare", shap_values, X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC We see that there is a linear relationship between how much people paid and their chance of survival. The richer they were the more likely they survived.  
# MAGIC Finally, let’s take a look at a few passengers more closely:

# COMMAND ----------

idx = 1
print(X_test.iloc[idx,:])
print(y_test[idx])

# COMMAND ----------

shap_display = shap.force_plot(explainer.expected_value, 
                               shap_values[idx], 
                               X_test.iloc[idx,:], 
                               #link="logit", 
                               matplotlib=True)
display(shap_display)

# COMMAND ----------

# MAGIC %md
# MAGIC That’s a plot of a passenger who didn’t survive. 
# MAGIC + The plot shows that his “Sex” (being male) and his “class” (being in the third class) were decreasing his survival rate. 
# MAGIC + The plot also shows that the number of siblings (“SibSp) being 0 increased his chance slightly. Maybe people who were alone in the ship without family were able to run faster without distraction.
# MAGIC
# MAGIC Let’s take a look at someone who survived:

# COMMAND ----------

idx = 4
print(X_test.iloc[idx,:])
print(y_test[idx])

# COMMAND ----------

shap_display = shap.force_plot(explainer.expected_value, 
                               shap_values[idx], 
                               X_test.iloc[idx], 
                               #link="logit", 
                               matplotlib=True)
display(shap_display)

# COMMAND ----------

# MAGIC %md
# MAGIC As expected, this person is female in class 1 who paid a high fare. This gave her a higher chance of survival. Also, the fact that she was a bit old (for the standard of time) decreased her chance a little bit.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 - Conclusion
# MAGIC The model interpretation allow us to try to explain what happened at the titanic.  
# MAGIC When the ship started to sink, rich people had the priority to leave the ship. Those with a fewer number of siblings were faster since they didn’t have to look for their family. When they found out that the lifeboats number was limited, they decided to prioritize children and women. So the priority was as the following: rich women and children, rich men, then everyone else. It’s very interesting how such insights can be fully extracted from a dataset.
