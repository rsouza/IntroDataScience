# Databricks notebook source
# MAGIC %md
# MAGIC # Nice Pipeline
# MAGIC 
# MAGIC here we present a nice example of a pipeline which we can use for training purposes. At first glance, it looks messy and hard to read.  
# MAGIC But if you take a moment to understand, you will notice the beauty for sure!

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer

# COMMAND ----------

# MAGIC %md
# MAGIC We just need to import some transformers which are inside of the pipeline.  
# MAGIC **This is not a operational code, just an example on longer pipelines.**

# COMMAND ----------

#Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

#Dimensionality reduction
from sklearn.decomposition import NMF

#Imputation
from sklearn.impute import SimpleImputer

#Modeling
from sklearn.ensemble import RandomForestClassifier

#Other
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Take a quick glance
# MAGIC Please take a quick look onto the pipeline which is below and come back here.
# MAGIC 
# MAGIC ## Step 2: Slow walkthrough
# MAGIC Get a **high level view** like this:
# MAGIC - look toward the top, there is a *FeatureUnion*, which is really a wrapper for entire feature engineering
# MAGIC - look at the bottom, there is a *RandomForestClassifier*, which is our predictive model
# MAGIC 
# MAGIC Now we can go deeper inside of our FeatureUnion, which is our **feature engineering**:
# MAGIC - it splits into three parts, depending on which features we are attempting to process
# MAGIC     - on top, we have numerical features
# MAGIC     - in the middle, we have categorical features
# MAGIC     - on the bottom, we have textual features
# MAGIC - now zoom out again and realize that this is wrapped under FeatureUnion, which means that these features will be transformed in a parallel way and appended next to each other
# MAGIC 
# MAGIC Only now let's **zoom into one part of our feature engineering**, for example into "numerical features", on the top:
# MAGIC - inside of it, we right away need ColumnTransformer as we want to specify for which columns certain transformation will be applied by name or by type
# MAGIC - now we could already be applying transformers, but remember that ColumnTransformer by default drops all untransformed columns, which would mean that if we want to apply some transformations sequentially we would not be able to
# MAGIC 
# MAGIC Finally, **get used to the indentation** (the whitespacing). Your code editor helps with this. Get used to this by clicking just behind the last visible character on the line where you are. For example go behing the last bracket on the line of *SimpleImputer*. Now if you hit Enter, it will land where a code should continue on the next line it you still want to stay within the element, which is the *Pipeline*.

# COMMAND ----------

# MAGIC %md
# MAGIC Source1: https://www.codementor.io/@bruce3557/beautiful-machine-learning-pipeline-with-scikit-learn-uiqapbxuj
# MAGIC Source2: http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html 

# COMMAND ----------

model_pipeline = Pipeline(steps=[
    ("features", FeatureUnion([
        ("numerical_features",
         ColumnTransformer([
             ("numerical",
              Pipeline(steps=[(
                  "impute_stage",
                  SimpleImputer(missing_values=np.nan, strategy="median")
              )]),
              ["feature_1"]
             )
         ])
        ), 
        ("categorical_features",
            ColumnTransformer([
                ("country_encoding",
                 Pipeline(steps=[
                     ("ohe", OneHotEncoder(handle_unknown="ignore")),
                     ("reduction", NMF(n_components=8)),
                 ]),
                 ["country"],
                ),
            ])
        ), 
        ("text_features",
         ColumnTransformer([
             ("title_vec",
              Pipeline(steps=[
                  ("tfidf", TfidfVectorizer()),
                  ("reduction", NMF(n_components=50)),
              ]),
              "title"
             )
         ])
        )
    ])
    ),
    ("classifiers", RandomForestClassifier())
])

# COMMAND ----------

# MAGIC %md
# MAGIC Now we would work with the pipeline easily:

# COMMAND ----------

#model_pipeline.fit(train_data, train_labels.values)
#predictions = model_pipeline.predict(predict_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. How to write that?
# MAGIC Alright, I now have a feeling that I am comfortable with understanding these, but how do we get to write such thing? The answer is: **from the outside - inwards**. Let's walk through an example, of course you could write things differently.  
# MAGIC 
# MAGIC At first, lay yourself a simple structure which separates your feature engineering (inside of FeatureUnion) and your predictive model.

# COMMAND ----------

model_pipeline = Pipeline(steps=[
    ("features", FeatureUnion([#all feature engineering goes here])),
    ("classifiers", RandomForestClassifier())
])

# COMMAND ----------

# MAGIC %md
# MAGIC Secondly, depending on your features, split yourself various parts inside of your feature engineering.

# COMMAND ----------

model_pipeline = Pipeline(steps=[
    ("features", FeatureUnion([("numerical_features", #numerical transformations), 
                               ("categorical_features", #categorical transformations), 
                               ("text_features", #textual transformations)
                              ])
    ),
    ("classifiers", RandomForestClassifier())
])

# COMMAND ----------

# MAGIC %md
# MAGIC Now you want to put inside a ColumnTransformer as the transformations will be applied only to specific columns.

# COMMAND ----------

model_pipeline = Pipeline(steps=[
    ("features", FeatureUnion([("numerical_features", ColumnTransformer([#numerical transformations])),
                               ("categorical_features", ColumnTransformer([#categorical transformations])),
                               ("text_features", ColumnTransformer([#textual transformations]))
                              ])
    ),
    ("classifiers", RandomForestClassifier())
])

# COMMAND ----------

# MAGIC %md
# MAGIC You can put Pipeline inside of it, for example, in case you have transformers which need to be sequential (such as numeric scaling and feature selection).  
# MAGIC And you just start to put in your individually wrote transformations from before.

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Reflect
# MAGIC Continue with this point only once you went through the pipeline above.  
# MAGIC 
# MAGIC Usually we think that nicely written code costs significantly more effort than code scraped together in whichever way. Now that we went through the composite estimators properly, you know that it might be even simpler in many cases, not to mention robustness.  
# MAGIC 
# MAGIC You are hopefully able to tell apart two things:  
# MAGIC - Data preprocessing and wrangling.
# MAGIC - Data preparation for ML (Feature Engineering)  
# MAGIC 
# MAGIC Always try to separate these things in your use case (code). That is why we present these topics separatedely. It will be of tremendous help in the longer run to write code in this way.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Working Example  
# MAGIC [Source](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

# COMMAND ----------

train = pd.read_csv("data_titanic/train.csv")
train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Use ``ColumnTransformer`` by selecting column by names
# MAGIC 
# MAGIC We will train our classifier with the following features:
# MAGIC 
# MAGIC Numeric Features:
# MAGIC 
# MAGIC * ``Age``: float;
# MAGIC * ``Fare``: float.
# MAGIC 
# MAGIC Categorical Features:
# MAGIC 
# MAGIC * ``Embarked``: categories encoded as strings ``{'C', 'S', 'Q'}``;
# MAGIC * ``Sex``: categories encoded as strings ``{'female', 'male'}``;
# MAGIC * ``Pclass``: ordinal integers ``{1, 2, 3}``.
# MAGIC 
# MAGIC We create the preprocessing pipelines for both numeric and categorical data.
# MAGIC Note that ``pclass`` could either be treated as a categorical or numeric
# MAGIC feature.

# COMMAND ----------

X = train.drop('Survived', axis=1)
y = train['Survived']

# COMMAND ----------

numeric_features = ["Age", "Fare"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), 
                                      ("scaler", StandardScaler())]
                              )

categorical_features = ["Embarked", "Sex", "Pclass"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                               ("cat", categorical_transformer, categorical_features),
                                              ]
                                )

# COMMAND ----------

# MAGIC %md
# MAGIC Append classifier to preprocessing pipeline. Now we have a full prediction pipeline.

# COMMAND ----------

clf = Pipeline(steps=[("preprocessor", preprocessor), 
                      ("classifier", LogisticRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf.fit(X_train, y_train)

print("model score: %.3f" % clf.score(X_test, y_test))

# COMMAND ----------

clf

# COMMAND ----------

# MAGIC %md
# MAGIC Use ``ColumnTransformer`` by selecting column by data types
# MAGIC 
# MAGIC When dealing with a cleaned dataset, the preprocessing can be automatic by
# MAGIC using the data types of the column to decide whether to treat a column as a
# MAGIC numerical or categorical feature.
# MAGIC 
# MAGIC `sklearn.compose.make_column_selector` gives this possibility.
# MAGIC 
# MAGIC <div class="alert alert-info"><h4>Note</h4><p>In practice, you will have to handle yourself the column data type.
# MAGIC    If you want some columns to be considered as `category`, you will have to
# MAGIC    convert them into categorical columns. If you are using pandas, you can
# MAGIC    refer to their documentation regarding [Categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html).</p></div>
# MAGIC 
# MAGIC 
# MAGIC + First, we will transform the object columns into categorical.  
# MAGIC + Then, let's only select a subset of columns to simplify our example.

# COMMAND ----------

X["Embarked"] = X["Embarked"].astype("category")
X["Sex"] = X["Sex"].astype("category")

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
subset_feature = ["Embarked", "Sex", "Pclass", "Age", "Fare"]
X_train, X_test = X_train[subset_feature], X_test[subset_feature]

# COMMAND ----------

X_train.info()

# COMMAND ----------

# MAGIC %md
# MAGIC We can observe that the `embarked` and `sex` columns were tagged as `category` columns.  
# MAGIC Therefore, we can use this information to dispatch the categorical columns to the ``categorical_transformer`` and the remaining columns to the ``numerical_transformer``.

# COMMAND ----------

from sklearn.compose import make_column_selector as selector

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category")),
    ]
)
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)


clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
clf

# COMMAND ----------

# MAGIC %md
# MAGIC The resulting score is not exactly the same as the one from the previous
# MAGIC pipeline because the dtype-based selector treats the ``pclass`` column as
# MAGIC a numeric feature instead of a categorical feature as previously:

# COMMAND ----------

selector(dtype_exclude="category")(X_train)

# COMMAND ----------

selector(dtype_include="category")(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Using the prediction pipeline in a grid search  
# MAGIC 
# MAGIC Grid search can also be performed on the different preprocessing steps defined in the ``ColumnTransformer`` object, together with the classifier's
# MAGIC hyperparameters as part of the ``Pipeline``.  
# MAGIC We will search for both the imputer strategy of the numeric preprocessing and the regularization parameter of the logistic regression using
# MAGIC :class:`~sklearn.model_selection.GridSearchCV`.

# COMMAND ----------

param_grid = {"preprocessor__num__imputer__strategy": ["mean", "median"],
              "classifier__C": [0.1, 1.0, 10, 100],
             }

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search

# COMMAND ----------

# MAGIC %md
# MAGIC Calling 'fit' triggers the cross-validated search for the best hyper-parameters combination:

# COMMAND ----------

grid_search.fit(X_train, y_train)

print("Best params:")
print(grid_search.best_params_)

# COMMAND ----------

# MAGIC %md
# MAGIC The internal cross-validation scores obtained by those parameters is:  

# COMMAND ----------

print(f"Internal CV score: {grid_search.best_score_:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC We can also introspect the top grid search results as a pandas dataframe:  

# COMMAND ----------

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)
cv_results[["mean_test_score",
            "std_test_score",
            "param_preprocessor__num__imputer__strategy",
            "param_classifier__C",
           ]].head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC The best hyper-parameters have be used to re-fit a final model on the full training set.  
# MAGIC We can evaluate that final model on held out test data that was not used for hyperparameter tuning.  

# COMMAND ----------

print(f"best logistic regression from grid search: {grid_search.score(X_test, y_test):.3f}")
