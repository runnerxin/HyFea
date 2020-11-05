#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgbm
from datetime import datetime
from time import gmtime, strftime
from scipy import stats
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error


# In[2]:


all_data = pd.read_csv('./data/feature_data_530.csv')


# glove
glove_tags = pd.read_csv('./data/alltags_feature.csv')
glove_title = pd.read_csv('./data/title_feature.csv')
all_data = pd.concat([all_data, glove_tags, glove_title], axis=1)


# In[3]:


train_all_data = all_data[all_data['train_type'] != -1]
submit_all_data = all_data[all_data['train_type'] == -1]

train_all_data = train_all_data.reset_index(drop=True)
submit_all_data = submit_all_data.reset_index(drop=True)

feature_columns = ['Pid', 'train_type', 'label', 'mean_label'] 
feature_columns += ['user_fe_{}'.format(i) for i in range(399)]
feature_columns += ['loc_fe_{}'.format(i) for i in range(400)]

train_label_df = train_all_data[['Pid', 'label']]
train_feature_df = train_all_data.drop(feature_columns, axis=1)

submit_label_df = submit_all_data[['Pid', 'label']]
submit_feature_df = submit_all_data.drop(feature_columns, axis=1)

print(len(train_feature_df), len(submit_feature_df))
print(len(train_label_df), len(submit_label_df))


# In[4]:


from catboost import CatBoostRegressor, Pool

cb_params = {
    'objective': 'RMSE',
    'eval_metric': 'MAE',
    'learning_rate': 0.03,
    'l2_leaf_reg': 3,
    'max_ctr_complexity': 1,
    'depth': 8,
    'leaf_estimation_method': 'Gradient',
    'use_best_model': True,
    'iterations': 100000,
    'early_stopping_rounds': 5000,
    'verbose': 500
}


cate_cols = ['Uid', 'Category', 'Subcategory', 'Concept', 'Mediatype', 'hour', 'day', 'weekday', 'week_hour', 'year_weekday',
             'Geoaccuracy', 'ispro' , 'Ispublic', 'img_model']
submit_data = Pool(data=submit_feature_df, label=submit_label_df['label'], cat_features=cate_cols)


valid_ans = []
submit_proba = []
kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
k = 0

for train_idx, valid_idx in kfold.split(train_feature_df, train_label_df):

    fold_valid_x, fold_valid_y = train_feature_df.loc[valid_idx], train_label_df['label'].loc[valid_idx]
    valid_data = Pool(data=fold_valid_x, label=fold_valid_y, cat_features=cate_cols)
    
    cb_model = CatBoostRegressor()
    cb_model.load_model('./save_model/KFold_catboost_' + str(k) + '.pkl')
    
    valid_pred = cb_model.predict(valid_data)
    valid_mse = mean_squared_error(fold_valid_y, valid_pred)
    valid_mae = mean_absolute_error(fold_valid_y, valid_pred)
    valid_src = stats.spearmanr(fold_valid_y, valid_pred)[0]
    print("MSE: %.4f, MAE: %.4f, SRC: %.4f"%(valid_mse, valid_mae, valid_src))
    valid_ans.append([valid_mse, valid_mae, valid_src])
        
    submit_pred = cb_model.predict(submit_data)
    submit_proba.append(submit_pred)
    
    k += 1

valid_ans = np.mean(valid_ans, axis=0)
print("valid: MSE: %.4f, MAE: %.4f, SRC: %.4f"%(valid_ans[0], valid_ans[1], valid_ans[2]))

# save result json
submit_ans = np.mean(submit_proba, axis=0)
result = pd.DataFrame()
result['post_id'] = submit_label_df['Pid'].apply(lambda x: 'post' + str(x))
result['popularity_score'] = submit_ans.round(decimals=4)

out_json = dict()
out_json["version"] = "VERSION 1.0"
out_json["result"] = result.to_dict(orient='records')
out_json["external_data"] = {"used": "true", "details": "catboost"}
f = open('KFold_catboost.json', "w")
json.dump(out_json, f)
f.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




