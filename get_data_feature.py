#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import pandas as pd
from datetime import datetime
from time import gmtime, strftime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import random
import os
import re
import json
import numpy as np
import pandas as pd
import lightgbm as lgbm
from datetime import datetime
from time import gmtime, strftime
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import lightgbm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

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


random_seed = 2020
random.seed(random_seed)
np.random.seed(random_seed)


# In[2]:


train_category = json.load(open('./data/train_all_json/train_category.json', encoding="utf-8"))
train_category_pd = pd.DataFrame(train_category)

train_additional = json.load(open('./data/train_all_json/train_additional.json', encoding="utf-8"))
train_additional_pd = pd.DataFrame(train_additional)

train_tags = json.load(open('./data/train_all_json/train_tags.json', encoding="utf-8"))
train_tags_pd = pd.DataFrame(train_tags)

train_temporalspatial = json.load(open('./data/train_all_json/train_temporalspatial.json', encoding="utf-8"))
train_temporalspatial_pd = pd.DataFrame(train_temporalspatial)

train_userdata = json.load(open('./data/train_all_json/train_userdata.json', encoding="utf-8"))
train_userdata_pd = pd.DataFrame(train_userdata)

train_img_pd = pd.read_csv('./data/train_all_json/train_img.txt', header=None)
train_img_pd.columns = ['img']
train_img_pd['img_file'] = train_img_pd['img'].apply(lambda x: './data/train/' + x[30:] + '.jpg')

train_label_pd = pd.read_csv('./data/train_all_json/train_label.txt', header=None)
train_label_pd.columns = ['label']

train_data = train_category_pd.merge(train_additional_pd, on=('Pid', 'Uid'), how='left')
train_data = train_data.merge(train_tags_pd, on=('Pid', 'Uid'), how='left')
train_data = train_data.merge(train_temporalspatial_pd, on=('Pid', 'Uid'), how='left')

train_data =  pd.concat([train_data, train_userdata_pd, train_img_pd, train_label_pd], axis=1)
print(len(train_data))
print(train_data.columns)


# In[3]:


test_category = json.load(open('./data/test_all_json/test_category.json', encoding="utf-8"))
test_category_pd = pd.DataFrame(test_category)

test_additional = json.load(open('./data/test_all_json/test_additional.json', encoding="utf-8"))
test_additional_pd = pd.DataFrame(test_additional)

test_tags = json.load(open('./data/test_all_json/test_tags.json', encoding="utf-8"))
test_tags_pd = pd.DataFrame(test_tags)

test_temporalspatial = json.load(open('./data/test_all_json/test_temporalspatial.json', encoding="utf-8"))
test_temporalspatial_pd = pd.DataFrame(test_temporalspatial)

test_userdata = json.load(open('./data/test_all_json/test_userdata.json', encoding="utf-8"))
test_userdata_pd = pd.DataFrame(test_userdata)

test_img_pd = pd.read_csv('./data/test_all_json/test_imgfile.txt', header=None)
test_img_pd.columns = ['img']
# test_img_pd['img_file'] = test_img_pd['img']
test_img_pd['img_file'] = test_img_pd['img']

test_data = test_category_pd.merge(test_additional_pd, on=('Pid', 'Uid'), how='left')
test_data = test_data.merge(test_tags_pd, on=('Pid', 'Uid'), how='left')
test_data = test_data.merge(test_temporalspatial_pd, on=('Pid', 'Uid'), how='left')

test_data =  pd.concat([test_data, test_userdata_pd, test_img_pd], axis=1)
test_data['label'] = -1

print(len(test_data))
print(test_data.columns)


# In[4]:



def pandas_split_valid_test_dataset(pandas_dataset, valid_ratio=0.1, test_ratio=0.1, shuffle=True):
    
    index = list(range(len(pandas_dataset)))
#     if shuffle:
#         random.shuffle(index)
    
    length = len(pandas_dataset)
    len_valid = int(length * valid_ratio + 0.6)
    len_test = int(length * test_ratio + 0.6)

    train_data = pandas_dataset.loc[index[:-len_test-len_valid]]
    valid_data = pandas_dataset.loc[index[-len_test-len_valid:-len_test]]
    test_data = pandas_dataset.loc[index[-len_test:]]
    return train_data, valid_data, test_data

train_df, valid_df, test_df = pandas_split_valid_test_dataset(train_data, valid_ratio=0.1, test_ratio=0.1, shuffle=True)
print(len(train_df), len(valid_df), len(test_df))


train_df['train_type'] = 0
valid_df['train_type'] = 1
test_df['train_type'] = 2
test_data['train_type'] = -1

all_data = pd.concat([train_df, valid_df, test_df, test_data], axis=0, sort=False)
all_data = all_data.reset_index(drop=True)
# print(len(all_data))
# all_data = all_data.fillna('0')

all_data.to_csv('./data/combine_data_530.csv', header=True)


# In[ ]:





# In[5]:


all_data = pd.read_csv('./data/combine_data_530.csv', low_memory=False)
all_data = all_data.fillna('0')


# In[6]:


glove_file ='./data/glove.42B.300d.txt'          # 已有的glove词向量
tmp_file = './data/word2vec.txt'                                              # 指定转化为word2vec格式后文件的位置
(count, dimensions) = glove2word2vec(glove_file, tmp_file)
print(count, dimensions)
print('glove2word2vec over')

# 加载转化后的文件  # 使用gensim载入word2vec词向量
wv_model = KeyedVectors.load_word2vec_format('./data/word2vec.txt')
print('load over')


# In[7]:




Alltags_split = all_data['Alltags'].apply(lambda x: x.lower().split(' '))
# Alltags_split = all_data['Alltags'].apply(lambda x: [w for w in re.sub('[^0-9a-zA-Z]', " ", x).lower().split(' ') if w != ""])

tags_ans = []
for sentence in Alltags_split:
    v = [wv_model[w] for w in sentence if w in wv_model]
    if len(v) == 0:
        tags_ans.append(np.zeros(300))
    else:
        tags_ans.append(np.mean(v, 0))

alltags_feature = np.array(tags_ans)

pd_alltags_feature = pd.DataFrame(alltags_feature, dtype='float')
pd_alltags_feature.columns = ['alltags_fe_{}'.format(i) for i in range(300)]
pd_alltags_feature.to_csv('./data/alltags_feature.csv', header=True, index=None)

print('alltag over!')


# In[8]:



Title_split = all_data['Title'].apply(lambda x: x.lower().split(' '))
# Title_split = all_data['Title'].apply(lambda x: [w for w in re.sub('[^0-9a-zA-Z]', " ", x).lower().split(' ') if w != ""])

title_ans = []
for sentence in Title_split:
    v = [wv_model[w] for w in sentence if w in wv_model]
    if len(v) == 0:
        title_ans.append(np.zeros(300))
    else:
        title_ans.append(np.mean(v, 0))

title_feature = np.array(title_ans)

pd_title_feature = pd.DataFrame(title_feature, dtype='float')
pd_title_feature.columns = ['title_fe_{}'.format(i) for i in range(300)]
pd_title_feature.to_csv('./data/title_feature.csv', header=True, index=None)

print('title over!')


# In[ ]:





# In[ ]:





# In[9]:


user_path = dict()
for i in range(len(all_data)):
    user = all_data['Uid'][i]
    path = all_data['Pathalias'][i]
    
    if user not in user_path:
        user_path[user] = set()
    if path != 'None':
        user_path[user].add(path)

Pathalias_list = []
for i in range(len(all_data)):
    user = all_data['Uid'][i]
    if len(user_path[user]) != 0:
        Pathalias_list.append(list(user_path[user])[0])
    else:
        Pathalias_list.append('None')
all_data['Pathalias'] = Pathalias_list


# In[10]:


user_additional = pd.read_csv('./data/user_additional.csv')
user_additional[user_additional['Pathalias'] == 'None'] = ['None', 0, 0, 0, 0, 0, 0, 0, 0]

all_data = pd.merge(all_data, user_additional, on='Pathalias', how='left')
all_data['label'] = all_data['label'].apply(lambda x: x if x!=-1 else 0)


# In[11]:


def get_img_data(img_file):
    if os.path.exists(img_file) == True:
        return img_file
    else:
        return './data/none_picture.jpg'

def get_feature(data_df):

    feature_data = pd.DataFrame()
    feature_data['Pid'] = data_df['Pid']
    feature_data['train_type'] = data_df['train_type']
    
    Uid_set=set(data_df['Uid'])
    Uid_map = dict(zip(Uid_set, list(range(len(Uid_set)))))
    feature_data['Uid'] = data_df['Uid'].map(Uid_map)
    
    feature_data['Uid_count'] = data_df['Uid'].map(dict(data_df.groupby('Uid')['Pid'].count()))
    feature_data['mean_label']= data_df['Uid'].map(dict(data_df.groupby('Uid')['label'].mean()))
    
    # Category
    Category_set=set(data_df['Category'])
    Category_map = dict(zip(Category_set, list(range(len(Category_set)))))
    feature_data['Category'] = data_df['Category'].map(Category_map)

    Subcategory_set=set(data_df['Subcategory'])
    Subcategory_map = dict(zip(Subcategory_set, list(range(len(Subcategory_set)))))
    feature_data['Subcategory'] = data_df['Subcategory'].map(Subcategory_map)
    
    Concept_set=set(data_df['Concept'])
    Concept_map = dict(zip(Concept_set, list(range(len(Concept_set)))))
    feature_data['Concept'] = data_df['Concept'].map(Concept_map)
    
    # title alltags base
    feature_data['Title_len'] = data_df['Title'].apply(lambda x: len(x))
    feature_data['Title_number'] = data_df['Title'].apply(lambda x: len(x.lower().split(' ')))
    feature_data['Alltags_len'] = data_df['Alltags'].apply(lambda x: len(x))
    feature_data['Alltags_number'] = data_df['Alltags'].apply(lambda x: len(x.lower().split(' ')))
    
    # img base
    data_df['img_file'] = data_df['img_file'].apply(lambda x: get_img_data('/home/ssd1/yhzhang/SMP2020/' + x))
    img_mode_map = {'P': 0, 'L': 1, 'RGB': 2, 'CMYK': 3}
    img_length, img_width, img_pixel, img_model = [], [], [], []
    for file in data_df['img_file']:
        pm = Image.open(file)
        img_length.append(pm.size[0])
        img_width.append(pm.size[1])
        img_pixel.append(pm.size[0] * pm.size[1])
        img_model.append(img_mode_map[pm.mode])
    feature_data['img_length'] = img_length
    feature_data['img_width'] = img_width
    feature_data['pixel'] = img_pixel
    feature_data['img_model'] = img_model
    
    # title svd
    tf_idf_enc_t = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_vec_t = tf_idf_enc_t.fit_transform(data_df['Title'])
    svd_enc_t = TruncatedSVD(n_components=20, n_iter=100, random_state=2020)
    mode_svd_t = svd_enc_t.fit_transform(tf_idf_vec_t)
    mode_svd_t = pd.DataFrame(mode_svd_t)
    mode_svd_t.columns = ['svd_mode_t_{}'.format(i) for i in range(20)]
    feature_data = pd.concat([feature_data, mode_svd_t], axis=1)
    
    # Tags svd
    tf_idf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_vec = tf_idf_enc.fit_transform(data_df['Alltags'])
    svd_enc = TruncatedSVD(n_components=20, n_iter=100, random_state=2020)
    mode_svd = svd_enc.fit_transform(tf_idf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_mode_{}'.format(i) for i in range(20)]
    feature_data = pd.concat([feature_data, mode_svd], axis=1)
    
    Mediatype_set=set(data_df['Mediatype'])
    Mediatype_map = dict(zip(Mediatype_set, list(range(len(Mediatype_set)))))
    feature_data['Mediatype'] = data_df['Mediatype'].map(Mediatype_map)
    
    # Temporal-spatial
    data_df['datetime'] = data_df['Postdate'].apply(lambda x: datetime.fromtimestamp(int(x)))
    feature_data['hour'] = data_df['datetime'].apply(lambda x: x.hour)
    feature_data['day'] = data_df['datetime'].apply(lambda x: x.day)
    feature_data['weekday'] = data_df['datetime'].apply(lambda x: x.weekday())
    feature_data['week_hour'] = data_df['datetime'].apply(lambda x: x.weekday() * 7 + x.hour)
    feature_data['year_weekday'] = data_df['datetime'].apply(lambda x: x.isocalendar()[1])
    
    feature_data['Longitude'] = data_df['Longitude'].apply(lambda x: float(x))
    feature_data['Latitude'] = data_df['Latitude'].apply(lambda x: float(x))
    
    feature_data['Geoaccuracy'] = pd.DataFrame(data_df['Geoaccuracy'], dtype='int')
    
    # User data
    feature_data['photo_count'] = pd.DataFrame(data_df['photo_count'], dtype='int')
    feature_data['ispro'] = pd.DataFrame(data_df['ispro'], dtype='int')
    
    user_fe = pd.DataFrame(np.array(list(data_df["user_description"].apply(lambda x: x.split(',')))), dtype='float')
    user_fe.columns = ['user_fe_{}'.format(i) for i in range(399)]
    
    loc_fe =pd.DataFrame(np.array(list(data_df["location_description"].apply(lambda x: x[:-2].split(',') if x[:-2] !='' else ['0.0']*400))), dtype='float')
    loc_fe.columns = ['loc_fe_{}'.format(i) for i in range(400)]
    feature_data = pd.concat([feature_data, user_fe, loc_fe], axis=1)
    
    photo_firstdate = data_df['photo_firstdate'].apply(lambda x: datetime.fromtimestamp(int(x) if x!='None' else 0))
    feature_data['firstdate'] = (data_df['datetime'] - photo_firstdate).apply(lambda x: x.days)
    feature_data['firstweek'] = feature_data['firstdate'] // 7
    feature_data['firstmonth'] = feature_data['firstdate'] // 30
    
    photo_firstdatetaken = data_df['photo_firstdatetaken'].apply(lambda x: datetime.fromtimestamp(int(x)))
    feature_data['firstdatetaken'] = (data_df['datetime'] - photo_firstdatetaken).apply(lambda x: x.days)
    feature_data['firstdatetakenweek'] = feature_data['firstdatetaken'] // 7
    feature_data['firstdatetakenmonth'] = feature_data['firstdatetaken'] // 30

    # Additional
    feature_data['totalViews'] = pd.DataFrame(data_df['totalViews'], dtype='int')
    feature_data['totalTags'] = pd.DataFrame(data_df['totalTags'], dtype='int')
    feature_data['totalGeotagged'] = pd.DataFrame(data_df['totalGeotagged'], dtype='int')
    feature_data['totalFaves'] = pd.DataFrame(data_df['totalFaves'], dtype='int')
    feature_data['totalInGroup'] = pd.DataFrame(data_df['totalInGroup'], dtype='int')
    feature_data['photoCount'] = pd.DataFrame(data_df['photoCount'], dtype='int')
    meanView, meanTags, meanFaves = [], [], []
    for i in range(len(data_df['photoCount'])):
        if data_df['photoCount'][i] == 0:
            meanView.append(0)
            meanTags.append(0)
            meanFaves.append(0)
        else:
            meanView.append(data_df['totalViews'][i] / data_df['photoCount'][i])
            meanTags.append(data_df['totalTags'][i] / data_df['photoCount'][i])
            meanFaves.append(data_df['totalFaves'][i] / data_df['photoCount'][i])
    feature_data['meanView'] = meanView
    feature_data['meanTags'] = meanTags
    feature_data['meanFaves'] = meanFaves
    feature_data['followerCount'] = pd.DataFrame(data_df['followerCount'], dtype='int')
    feature_data['followingCount'] = pd.DataFrame(data_df['followingCount'], dtype='int')
    
    Ispublic_set=set(data_df['Ispublic'])
    Ispublic_map = dict(zip(Ispublic_set, list(range(len(Ispublic_set)))))
    feature_data['Ispublic'] = data_df['Ispublic'].map(Ispublic_map)

    # label
    feature_data['label'] = pd.DataFrame(data_df['label'], dtype='float')
    return feature_data


save_feature_df = get_feature(all_data)
save_feature_df.to_csv('./data/feature_data_530.csv', header=True, index=None)
print('feature save!')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




