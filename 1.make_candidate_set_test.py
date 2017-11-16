import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

dir = './data/'

user_shop_behavior = pd.read_csv(dir + '/ccf_first_round_user_shop_behavior.csv').reset_index()
train_wifi = pd.read_csv(dir + '/train_wifi.csv')
train_wifi = pd.merge(train_wifi,user_shop_behavior[['index','time_stamp']])
del user_shop_behavior
print(train_wifi.head())

# offline_val = train_wifi[(train_wifi['time_stamp'] < '2017-09-01 00:00')&(train_wifi['time_stamp'] >= '2017-08-25 00:00' )]
# # print(offline_val.head())
# offline_val_feature = train_wifi[(train_wifi['time_stamp'] < '2017-08-25 00:00')&(train_wifi['time_stamp'] >= '2017-08-11 00:00' )]

print('offline_train_set')
train_wifi_train = train_wifi[(train_wifi['time_stamp'] < '2017-08-25 00:00')&(train_wifi['time_stamp'] >= '2017-08-11 00:00' )]
print(train_wifi_train.shape)
train_wifi_test = train_wifi[(train_wifi['time_stamp'] < '2017-09-01 00:00')&(train_wifi['time_stamp'] >= '2017-08-25 00:00' )]
print(train_wifi_test.shape)

# 选择构造特征的候选集的数量 n = 1 2 3 .
train_wifi_train_candidate = train_wifi_train.sort_values(['index','strength'],ascending=False).groupby(['index'],as_index=False).head(2)
# 选择候选集中出现bssid - shopid 的匹配频次
train_wifi_train_candidate = train_wifi_train_candidate.groupby(['bssid','shop_id'],as_index=False).strength.agg({'bsCount':'count'}).sort_values(['bssid','bsCount'],ascending=False)
train_wifi_train_candidate = train_wifi_train_candidate.groupby(['bssid'],as_index=False).head(6)

# 选择训练数据供候选的bassid
train_wifi_test_candidate = train_wifi_test.sort_values(['index','strength'],ascending=False).groupby(['index'],as_index=False).head(3)

most_prop = pd.merge(train_wifi_train_candidate ,train_wifi_test_candidate ,how='right',on='bssid')

# 制作标签
most_prop['yes']= most_prop.shop_id_y == most_prop.shop_id_x
print(most_prop.sort_values(['index','yes'],ascending=False))

most_prop_ = most_prop.sort_values(['index','yes'],ascending=False).groupby(['index'],as_index=False).head(1)

print(most_prop_.yes.mean())
print(len(most_prop[['index','shop_id_x']].drop_duplicates()) / len(train_wifi_test.drop_duplicates(['index'])))

