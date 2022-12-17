#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[19]:


## libraries for reading and manipulating data
import pandas as pd
import numpy as np

## libraries for data visualization
import matplotlib.pyplot as plt

## libraries for computing accuracy score
from sklearn.metrics import (f1_score, accuracy_score, recall_score, precision_score,
                            confusion_matrix, confusion_matrix)

from sklearn import metrics

## library for data scaling
from sklearn.preprocessing import StandardScaler

## library for imputing missing values
from sklearn.impute import SimpleImputer

## model 
import xgboost
from xgboost import XGBClassifier

## library for 
from skops import hub_utils, card


##
import pickle
from pathlib import Path

## suppress warnings 
import warnings
warnings.filterwarnings("ignore")


# In[2]:


## loading dataset with pandas
train_generator_data = pd.read_csv("dataset/Train.csv.csv")
test_generator_data = pd.read_csv("dataset/Test.csv.csv")


# In[3]:


## creating a copy of the data
train_df = train_generator_data.copy()
test_df = test_generator_data.copy()


# In[4]:


## define a function to compute difference metrics score
## to check the performance of the classification model
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classifcation model performance
    
    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    ## generate predictions using independent variables
    pred = model.predict(predictors)

    ## calculating the accuracy
    acc = accuracy_score(target, pred)
    ## calculating the Recall
    recall = recall_score(target, pred)
    ## calculating the Precision
    precision = precision_score(target, pred)
    ## calculating the F1-Score
    f1 = f1_score(target, pred)

    ## creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
        }, 
        index=[0],
    )

    return df_perf


# In[5]:


def confusion_matrix(model, predictors, target):
    """
    function to plot the confusion matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variables  
    """
    y_valid_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred_valid)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2,2)

    plt.figure(figsize=(6,4))
    plt.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    


# In[6]:


## type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)


# ## Training Of Final Model

# In[7]:


## prepare the dataset

## create an instance of the imputer 
imputer = SimpleImputer(strategy="median")

## fit and transform the train dataframe
train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)

## fit and transform the test dataframe
test_df = pd.DataFrame(imputer.fit_transform(test_df), columns=train_df.columns)


# In[8]:


## lets select the target variable 
y_train = train_df['Target'].values
y_test = test_df['Target'].values

## delete the target variable 
del train_df['Target']
del test_df['Target']

## convert the dataframe to numpy array 
X_train = train_df.values
X_test = test_df.values


# In[9]:


## lets initialize our sandard scaler
stdc = StandardScaler()

## lets fit and transform
X_train = stdc.fit_transform(X_train)
X_test = stdc.transform(X_test)


# In[10]:


## build the pipeline to train the final model
xgboost_model_final =  XGBClassifier(
    random_state=1,
    eval_metric='logloss',
    subsample=0.8, 
    scale_pos_weight=10, 
    n_estimators=250,
    learning_rate=0.1,
    gamma=3)

xgboost_model_final.fit(X_train, y_train)


# ## Testing Final Model

# In[11]:


## lets test our final model 
xgboost_test_perf = model_performance_classification_sklearn(xgboost_model_final, X_test, y_test)
xgboost_test_perf


# ## Saving Final Model

# In[14]:


## lets save the model to a file
model_filename = 'model.pkl'

with open(model_filename, mode='bw') as f:
    pickle.dump(xgboost_model_final, file=f)


# ## Uploading The Model To HuggingFace

# In[21]:


local_repo = 'model_local_repo'
hub_utils.init(
    model=model_filename,
    requirements=[f"xgboost={xgboost.__version__}"],
    dst=local_repo,
    task='tabular-classification',
    data=X_test,
)


# In[22]:


## lets add a model card to the local model

model_card = card.Card(xgboost_model_final, metadata=card.metadata_from_config(Path(local_repo)))

# you can add several sections and metadata to the model card
model_card.add(
    model_card_authors= 'Moro abdul Wahab',
    model_description= 'ML classification model to predict or identify failures in a generator',
)

# save model card locally
model_card.save(Path(local_repo) / "README.md")


# In[23]:


## lets push our local model repo to the remote model repo on the Hub 
# if the repository doesn't exist remotely on the Hugging Face Hub,
# it will be created when we set create_remote to True
repo_id = "moro23/ml-generation-failure-prediction"
hub_utils.push(
    repo_id=repo_id,
    source=local_repo,
    token="###", # your personal token to be downloaded from hugging face
    commit_message="uploaded the first version of the generator failure prediction model.",
    create_remote=True,
)

