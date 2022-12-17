#!/usr/bin/env python
# coding: utf-8

# In[1]:


## import our libraries
## library for 
from skops import hub_utils

import pickle
from pathlib import Path

## suppress warnings 
import warnings
warnings.filterwarnings("ignore")


# In[6]:


# it will be created when we set create_remote to True
repo_id = "moro23/ml-generation-failure-prediction"

## create a repo on Hub
local_repo = "downloaded_model_from_hub"

model = hub_utils.download(repo_id=repo_id, dst=local_repo)
print(f'Loaded model: {model}')


# In[ ]:




