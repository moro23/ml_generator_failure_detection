#!/usr/bin/env python
# coding: utf-8

# In[2]:


## import our libraries
## library for 
from skops import hub_utils

import pickle
from pathlib import Path
import pandas as pd


from flask import Flask, request, jsonify


## suppress warnings 
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# it will be created when we set create_remote to True
repo_id = "moro23/ml-generation-failure-prediction"

## create a repo on Hub
local_repo = "downloaded_model_from_hub"

#model = hub_utils.download(repo_id=repo_id, dst=local_repo)
#print(f'Loaded model: {model}')


# In[4]:


## lets load the downloaded model into our program 
with open('generator_failure_prediction_model.pkl', 'rb') as f_in:
    stdc, model = pickle.load(f_in)


# In[5]:


## lets create a function generating the prediction
def predict_generator_failure(generator_sample,stdc, model):
    df = pd.DataFrame([generator_sample])
    X = stdc.transform(df)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred


## lets create a function generating the prediction
def predict_generator_failure_single(generator_sample,stdc, model):
    df = pd.DataFrame([generator_sample])
    X = stdc.transform(df)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred[0]

# In[6]:



# In[7]:
## lets create an instance of the Flask app
app = Flask('generator')

@app.route('/predict', methods=['POST'])
def predict():
    generator_sample = request.get_json()

    prediction = predict_generator_failure_single(generator_sample, stdc, model)
    print(f'Prediction: {round(prediction, 2)}')

    if prediction >= 0.5:
        return jsonify(f'Prediction: {round(prediction,2)}, Verdict: Failure')
    else:
        return jsonify(f'Prediction: {round(prediction,2)},Verdict: No Failure')



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)


