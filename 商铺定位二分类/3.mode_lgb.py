#coding:utf-8
import gc
import pandas as pd
import numpy as np

# 对nan不处理版本
# def customs_acc(pred_probs,dtrain):
#     label = dtrain.get_label()
#     preds = pred_probs > 0.5
#     correct = np.array(preds) & np.array(label)
#     precise = np.sum(correct) / len(correct)
#     return "accurary",-precise,False

# 根据经纬度计算距离
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
# 哈夫曼距离
def hafuman_km(lon1, lat1, lon2, lat2):
    return haversine_np(lon1,lat1,lon2,lat1) + haversine_np(lon2,lat1,lon2,lat2)

dir = './data/'
print('reading train/val data ')
train = pd.read_csv(dir + 'offline_train.csv')
val = pd.read_csv(dir + 'offline_val.csv')


shop_info_tmp = pd.read_csv(dir + 'ccf_first_round_shop_info.csv')
shop_info = shop_info_tmp[['shop_id','category_id','longitude','latitude','price']]
del shop_info_tmp;gc.collect()
shop_info.rename(columns={'longitude':'s_longitude','latitude':'s_latitude'},inplace=True)

train.rename(columns={'shop_id_x':'shop_id'},inplace=True)
val.rename(columns={'shop_id_x':'shop_id'},inplace=True)

user_info_tmp = pd.read_csv(dir + 'ccf_first_round_user_shop_behavior.csv').reset_index()
user_info = user_info_tmp[['index','longitude','latitude']]
del user_info_tmp;gc.collect()

# 暴力去掉nan的数据
train = pd.merge(train,shop_info,on=['shop_id'],how='left')
train = train.dropna()
train = pd.merge(train,user_info,on=['index'],how='left')

train['distance'] = hafuman_km(train['s_longitude'],train['s_latitude'],train['longitude'],train['latitude'])
train['distance'] = np.log1p(train['distance'])
train['category_id'] = train['category_id'].map(lambda x:str(x).split('_')[1])
train['mall_id'] = train['mall_id'].map(lambda x:str(x).split('_')[1])
#
print(train.head())
print(train.columns)
# 特征字段
# Index([u'bssid', u'shop_id', u'bsCount', u'max_average', u'std_wifi',
       # u'sw_average', u'shop_around_bssid', u'shop_around_bssid_pinci',
       # u'wifi_around_shop', u'wifi_around_shop_pimco',
       # u'connect_shop_bssid_time', u'connect', u'index', u'mall_id',
       # u'shop_id_y', u'strength', u'time_stamp', u'label', u'category_id',
       # u's_longitude', u's_latitude', u'price', u'longitude', u'latitude',
       # u'distance']

val = pd.merge(val,shop_info,on=['shop_id'],how='left')
val = val.dropna()
val = pd.merge(val,user_info,on=['index'],how='left')
del user_info;gc.collect()
val['distance'] = hafuman_km(val['s_longitude'],val['s_latitude'],val['longitude'],val['latitude'])
val['distance'] = np.log1p(val['distance'])
val['category_id'] = val['category_id'].map(lambda x:str(x).split('_')[1])
val['mall_id'] = val['mall_id'].map(lambda x:str(x).split('_')[1])

print(val.head())
# Index(['bssid', 'connect', 'index', 'mall_id', 'shop_id', 'strength',
#        'time_stamp', 'min_average', 'max_average', 'sw_average', 'std_wifi',
#        'c_std_wifi', 'c_sw_average', 'c_min_average', 'c_max_average',
#        'b_min_average', 'b_max_average', 'std_b_wifi', 'bw_average',
#        'c_std_b_wifi', 'c_bw_average', 'c_b_max_average', 'c_b_min_average',
#        'shop_around_bssid', 'shop_around_bssid_pinci', 'wifi_around_shop',
#        'wifi_around_shop_pimco', 'connect_shop_bssid_time']

features = ['bsCount','connect','strength','category_id','distance','mall_id','price','min_average', 'max_average', 'sw_average', 'std_wifi',
       'c_std_wifi', 'c_sw_average', 'c_min_average', 'c_max_average','b_min_average', 'b_max_average', 'std_b_wifi', 'bw_average',
       'c_std_b_wifi', 'c_bw_average', 'c_b_max_average', 'c_b_min_average','shop_around_bssid', 'shop_around_bssid_pinci', 'wifi_around_shop',
       'wifi_around_shop_pimco', 'connect_shop_bssid_time']

train_train_label = train.pop('label')
train_val_label = val.pop('label')

import lightgbm as lgb
# load or create your dataset

y_train = train_train_label.values
y_test = train_val_label.values
X_train = train[features].values
X_test = val[features].values

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 256,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    # 'random_state':1024
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1500,
                # fobj=mape_object,
                # feval=customs_acc,
                valid_sets=lgb_eval,
                early_stopping_rounds=15)

result_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# print(preds_this_mall)

result_1 = pd.DataFrame({'pre_p':list(result_test)})
result_1 = pd.concat([val[['shop_id','index','shop_id_y']],result_1],axis=1)
result_1 = pd.DataFrame(result_1).sort_values('pre_p',ascending=False).drop_duplicates(['index'])
print(sum(result_1['shop_id'] == result_1['shop_id_y']) * 1.0 / len(result_1['shop_id']))
del train;gc.collect()
del val;gc.collect()
del result_1;gc.collect()

# save model to file
gbm.save_model('model.txt')


# # 提交数据
# sub = pd.read_csv(dir + 'sub_wifi.csv')
# print(sub.shape)
# sub_user_info = pd.read_csv(dir + 'evaluation_public.csv')
# # row_id,user_id,mall_id,time_stamp,longitude,latitude,wifi_infos
# sub_user_info = sub_user_info[['row_id','longitude','latitude']]
# sub = pd.merge(sub,shop_info,on=['shop_id'],how='left')
# del sub['label']
# del shop_info;gc.collect()
# # 暴力删除1万个nan数据
# sub = sub.dropna()
# sub = pd.merge(sub,sub_user_info,on=['row_id'],how='left')
# sub['distance'] = hafuman_km(sub['s_longitude'],sub['s_latitude'],sub['longitude'],sub['latitude'])
# sub['distance'] = np.log1p(sub['distance'])
# sub['category_id'] = sub['category_id'].map(lambda x:str(x).split('_')[1])
# sub['mall_id'] = sub['mall_id'].map(lambda x:str(x).split('_')[1])
# print(sub.head())
# print(sub.columns)
# sub_lgb = sub[features].values

# result = gbm.predict(sub_lgb, num_iteration=gbm.best_iteration)
# result = pd.DataFrame({'pre_p':list(result)})
# result = pd.concat([sub,result],axis=1)
# result = pd.DataFrame(result).sort_values('pre_p',ascending=False).drop_duplicates('row_id')
# print(result.shape)
# result[['row_id','shop_id','pre_p']].to_csv('./tmp.csv',index=None)
