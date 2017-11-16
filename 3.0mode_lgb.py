#coding:utf-8
import gc
import pandas as pd
import numpy as np
from tools import bearing_array,hafuman_km,get_features_list
# 对nan不处理版本
# def customs_acc(pred_probs,dtrain):
#     label = dtrain.get_label()
#     preds = pred_probs > 0.5
#     correct = np.array(preds) & np.array(label)
#     precise = np.sum(correct) / len(correct)
#     return "accurary",-precise,False

def naive_auc(labels,preds):
    """
    最简单粗暴的方法
　　　先排序，然后统计有多少正负样本对满足：正样本预测值>负样本预测值, 再除以总的正负样本对个数
     复杂度 O(NlogN), N为样本数
    """
    preds = preds.get_label()
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total_pair = n_pos * n_neg

    labels_preds = zip(labels,preds)
    labels_preds = sorted(labels_preds,key=lambda x:x[1])
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(len(labels_preds)):
        if labels_preds[i][0] == 1:
            satisfied_pair += accumulated_neg
        else:
            accumulated_neg += 1

    return 'auc',satisfied_pair / float(total_pair),False



features = get_features_list()

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

train['time_stamp'] = pd.to_datetime(train['time_stamp'])
train['current_hour'] =  pd.DatetimeIndex(train.time_stamp).hour
# train['current_week'] =  pd.DatetimeIndex(train.time_stamp).dayofweek

train['distance'] = hafuman_km(train['s_longitude'],train['s_latitude'],train['longitude'],train['latitude'])
train['current_bearing_array'] = bearing_array(train.s_latitude.values, train.s_longitude.values, 
                                                                                    train.latitude.values, train.longitude.values)  
# train['distance'] = np.log1p(train['distance'])

# 由于历史记录的意义并没有那么明确，因此将历史转化为和当前的差值处理
train['c_wifi_var'] = train['strength'] - train['c_sw_average']
train['wifi_var'] = train['strength'] - train['s_avg_power']

train['s_wifi_var'] = train['strength'] - train['w_avg_power']
train['sb_wifi_var'] = train['strength'] - train['sb_history_avg_power']


train['angle_var'] = train['history_bearing_array_median'] - train['current_bearing_array']
train['c_sb_wifi_var'] = train['strength'] - train['c_sb_history_avg_power']

train['s_sb_wifi_var_ratio'] = train['c_sb_wifi_var'] / (train['sb_wifi_var']+0.0001)
train['category_id'] = train['category_id'].map(lambda x:str(x).split('_')[1])
train['mall_id'] = train['mall_id'].map(lambda x:str(x).split('_')[1])
train['shop_id_f'] = train['shop_id'].map(lambda x:str(x).split('_')[1])
train['bssid_f'] = train['bssid'].map(lambda x:str(x).split('_')[1])

# train['sw_average_ratio'] = (train['c_sw_average'] + 0.5 ) / (train['sw_average'] + 1)
# train['bw_average_ratio'] = (train['c_bw_average'] + 0.5 )/ (train['bw_average'] + 1)

# train['price'] = np.log1p(train['price'])
#
print(train.head())
print(train.columns)
# 特征字段
train_train_label = train.pop('label')
y_train = train_train_label.values
X_train = train[features].values
del train;gc.collect()
del train_train_label;gc.collect()
gc.collect()


val = pd.merge(val,shop_info,on=['shop_id'],how='left')
val = val.dropna()
val = pd.merge(val,user_info,on=['index'],how='left')

val['time_stamp'] = pd.to_datetime(val['time_stamp'])
val['current_hour'] =  pd.DatetimeIndex(val.time_stamp).hour
# val['current_week'] =  pd.DatetimeIndex(val.time_stamp).dayofweek

del user_info;gc.collect()
val['distance'] = hafuman_km(val['s_longitude'],val['s_latitude'],val['longitude'],val['latitude'])
val['current_bearing_array'] = bearing_array(val.s_latitude.values, val.s_longitude.values, 
                                                                                    val.latitude.values, val.longitude.values)  
#val['distance'] = np.log1p(val['distance'])

val['c_wifi_var'] = val['strength'] - val['c_sw_average']
val['wifi_var'] = val['strength'] - val['s_avg_power']
val['s_wifi_var'] = val['strength'] - val['w_avg_power']
val['sb_wifi_var'] = val['strength'] - val['sb_history_avg_power']
val['angle_var'] = val['history_bearing_array_median'] - val['current_bearing_array']
val['c_sb_wifi_var'] = val['strength'] - val['c_sb_history_avg_power']

val['s_sb_wifi_var_ratio'] = val['c_sb_wifi_var'] / (val['sb_wifi_var']+0.0001)
val['category_id'] = val['category_id'].map(lambda x:str(x).split('_')[1])
val['mall_id'] = val['mall_id'].map(lambda x:str(x).split('_')[1])
val['shop_id_f'] = val['shop_id'].map(lambda x:str(x).split('_')[1])
val['bssid_f'] = val['bssid'].map(lambda x:str(x).split('_')[1])

# val['sw_average_ratio'] = (val['c_sw_average'] + 0.5 ) / (val['sw_average'] + 1)
# val['bw_average_ratio'] = (val['c_bw_average'] + 0.5 ) / (val['bw_average'] + 1)

# val['price'] = np.log1p(val['price'])

print(val.head())

train_val_label = val.pop('label')
y_test = train_val_label.values
X_test = val[features].values
del train_val_label;gc.collect()

import lightgbm as lgb
# load or create your dataset
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
del X_train;gc.collect()
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
                num_boost_round=1800,
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
# del train;gc.collect()
del val;gc.collect()
del result_1;gc.collect()

# save model to file
gbm.save_model('model.txt')


