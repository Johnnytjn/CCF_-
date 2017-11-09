#coding:utf-8
#1 .|8月18日-31日的候选特征|9月提交数据|
#2 .|8月11日-24日的候选特征|8月25日-8月31日构造线下验证|
#3 .|8月4日-17日的候选特征|8月18日-8月24日构造线下训练数据|

from tools import hafuman_km,mode_function
import math

import numpy as np
import pandas as pd
import gc
dir = './data/'

user_shop_behavior = pd.read_csv(dir + '/ccf_first_round_user_shop_behavior.csv').reset_index()
train_wifi = pd.read_csv(dir + '/train_wifi.csv')
train_wifi = pd.merge(train_wifi,user_shop_behavior[['index','time_stamp']])
del user_shop_behavior;gc.collect()
# print(train_wifi.head())
# 候选特征
def make_features(feature,org_data,mode= list([]),index='index',row_id='index'):
    # 选择构造特征的候选集的数量 n = 1 2 3 .


    # ps：自己提取啊啊啊啊啊啊啊
    # 这个部分大家可以自己筛选，我是先选择强度，后选择频次
    # 制作标签

    # 制作标签
    most_prop['connect'] = most_prop['connect'].astype(int)
    if row_id == 'row_id':
        most_prop['label'] = np.nan
    else:
        most_prop['label'] = most_prop.shop_id_y == most_prop.shop_id_x
        most_prop['label'] = most_prop['label'].astype(int)
    return most_prop
# 自己制作特征
def get_one_feature(count_feature):
    dir = './data/'  
    user_behavier = pd.read_csv(dir + 'ccf_first_round_user_shop_behavior.csv')[['longitude','latitude']].reset_index()
    shop_info_tmp = pd.read_csv(dir + 'ccf_first_round_shop_info.csv')[['shop_id','price','longitude','latitude','category_id']]
    # 加入类别
    shop_info_tmp['category_id'] = shop_info_tmp['category_id'].map(lambda x:str(x).split('_')[1])

    shop_info_tmp.rename(columns={'latitude':'s_latitude','longitude':'s_longitude'},inplace=True)
    count_feature_with_shop_price = pd.merge(count_feature,shop_info_tmp,on=['shop_id'],how='left')
    count_feature_with_shop_price_position = pd.merge(count_feature_with_shop_price,user_behavier,on=['index'],how='left')
    del user_behavier;gc.collect()
    del count_feature_with_shop_price;gc.collect()
    # 历史上的距离特征
    count_feature_with_shop_price_position['distance_history'] = hafuman_km(count_feature_with_shop_price_position['s_longitude'],count_feature_with_shop_price_position['s_latitude'],
    																		count_feature_with_shop_price_position['longitude'],count_feature_with_shop_price_position['latitude'])

    # 1 shop 的行为范围中位数（测试发现中位数效果好）
    shop_scale = count_feature_with_shop_price_position.groupby(['mall_id','shop_id'],as_index=False).distance_history.agg({'s_median_scale':np.median})
    count_feature = pd.merge(count_feature,shop_scale,on=['mall_id','shop_id'],how='left')
    del shop_scale;gc.collect()
    # 强度特征 均值
    # 2.1.历史商店周围的平均wifi强度 
    shop_around_wifi_power = count_feature_with_shop_price_position.groupby(['mall_id','shop_id'],as_index=False).strength.agg({'s_avg_power':np.mean})
    count_feature = pd.merge(count_feature,shop_around_wifi_power,on=['mall_id','shop_id'],how='left')
    del shop_around_wifi_power;gc.collect()

    # 2.1.1 发生链接时的商铺周围的wifi平均强度
    number_count_shop_wifi_strength_c = count_feature[count_feature['connect'] == 1].groupby(['mall_id','shop_id'],as_index=False).strength.agg({'c_sw_average':np.mean})
    count_feature = pd.merge(count_feature,number_count_shop_wifi_strength_c,on=['mall_id','shop_id'],how='left')
    del number_count_shop_wifi_strength_c;gc.collect()

    # 2.2.历史商店和wifi组合时，周围的wifi强度
    shop_bssid_around_wifi_power = count_feature_with_shop_price_position.groupby(['mall_id','bssid','shop_id'],as_index=False).strength.agg({'sb_history_avg_power':np.mean})
    count_feature = pd.merge(count_feature,shop_bssid_around_wifi_power,on=['mall_id','shop_id','bssid'],how='left')
    del shop_bssid_around_wifi_power;gc.collect()
    
    
    # 2.3.历史商店和wifi组合时且连接时的wifi，周围的wifi强度
    shop_bssid_around_wifi_power_c = count_feature_with_shop_price_position[count_feature_with_shop_price_position['connect'] == 1].groupby(['mall_id','bssid','shop_id'],as_index=False).strength.agg({'c_sb_history_avg_power':np.mean})
    count_feature = pd.merge(count_feature,shop_bssid_around_wifi_power_c,on=['mall_id','shop_id','bssid'],how='left')
    del shop_bssid_around_wifi_power_c;gc.collect()
    
    del count_feature_with_shop_price_position;gc.collect()

    # 2.4 商场中 wifi 的强度特征
    wifi_power_feat = count_feature.groupby(['mall_id','bssid'],as_index=False).strength.agg({'w_avg_power':np.mean,'w_std_power':np.std})
    count_feature = pd.merge(count_feature,wifi_power_feat,on=['mall_id','bssid'],how='left')
    del wifi_power_feat;gc.collect()

    # 时间串信息组合
    count_feature['time_stamp'] = pd.to_datetime(count_feature['time_stamp'])
    count_feature['history_hour'] =  pd.DatetimeIndex(count_feature.time_stamp).hour
    count_feature['history_day'] =  pd.DatetimeIndex(count_feature.time_stamp).day

    day_of_list = max(list(count_feature['history_day'].unique()))
    # print(day_of_list)
    # 构造时间推移特征值
    for day_index in [1,3,5,7]:
        day_count_feature = count_feature[count_feature.history_day >= (day_of_list - day_index + 1)]
        day_count_feature_shop_hot = day_count_feature.groupby(['mall_id','shop_id'],as_index=False).strength.count()
        day_count_feature_shop_hot.rename(columns={'strength':'%d_shop_hot'%(day_index)},inplace=True)
        count_feature = pd.merge(count_feature,day_count_feature_shop_hot,on=['mall_id','shop_id'],how='left')
        count_feature = count_feature.fillna(0)
    
    # 链接时发生的特征

    # 3.2 wifi被链接的次数
    wifi_is_connected_times = count_feature[count_feature['connect'] == 1].groupby(['mall_id','bssid'],as_index=False).strength.count()
    wifi_is_connected_times.rename(columns={'strength':'wifi_is_connected_times'},inplace=True)
    count_feature = pd.merge(count_feature,wifi_is_connected_times,on=['mall_id','bssid'],how='left')
    del wifi_is_connected_times;gc.collect()

    # 3.3 wifi被链接时与商铺发生的次数
    wifi_is_connected_shop_times = count_feature[count_feature['connect'] == 1].groupby(['mall_id','bssid','shop_id'],as_index=False).strength.count()
    wifi_is_connected_shop_times.rename(columns={'strength':'wifi_is_connected_shop_times'},inplace=True)
    count_feature = pd.merge(count_feature,wifi_is_connected_shop_times,on=['mall_id','shop_id','bssid'],how='left')
    del wifi_is_connected_shop_times;gc.collect()

    count_feature['shop_wifi_connect_ratio'] = count_feature['wifi_is_connected_shop_times'] / (count_feature['wifi_is_connected_times'] + 1.0 )

    # 3.wifi覆盖的shop个数
    wifi_cover_count = count_feature.groupby(['mall_id','bssid'],as_index=False).shop_id.apply(lambda x : len(set(x))).reset_index()
    wifi_cover_count.rename(columns={0:'wifi_cover_shop'},inplace=True)
    count_feature = pd.merge(count_feature,wifi_cover_count,on=['mall_id','bssid'],how='left')
    del wifi_cover_count;gc.collect()

    # tfidf-特征 wifi 的tfidf统计特征

    # 3.4 wifi和shop出现的次数
    wifi_shop_count = count_feature.groupby(['mall_id','shop_id','bssid'],as_index=False).strength.count()
    wifi_shop_count.rename(columns={'strength':'wifi_shop_count'},inplace=True)
    count_feature = pd.merge(count_feature,wifi_shop_count,on=['mall_id','shop_id','bssid'],how='left')
    del wifi_shop_count;gc.collect()

    # 3.5 shop有关的wifi个数
    wifi_shop_length = count_feature.groupby(['mall_id','shop_id'],as_index=False).bssid.count()
    wifi_shop_length.rename(columns={'bssid':'wifi_shop_length'},inplace=True)
    count_feature = pd.merge(count_feature,wifi_shop_length,on=['mall_id','shop_id'],how='left')
    del wifi_shop_length;gc.collect()

    count_feature['wifi_shop_ratio_tfidf'] = count_feature['wifi_shop_count'] / (count_feature['wifi_shop_length'] + 1.0)

    # 3.6 bssid个数
    mall_wifi_count = count_feature.groupby(['mall_id','bssid'],as_index=False).strength.count()
    mall_wifi_count.rename(columns={'strength':'mall_wifi_count'},inplace=True)
    count_feature = pd.merge(count_feature,mall_wifi_count,on=['mall_id','bssid'],how='left')
    del mall_wifi_count;gc.collect()

    # 3.7 商铺周围bssid的个数
    shop_around_count = count_feature.groupby(['mall_id','shop_id'],as_index=False).bssid.apply(lambda x : len(set(x))).reset_index()
    shop_around_count.rename(columns={0:'shop_around_count'},inplace=True)
    count_feature = pd.merge(count_feature,shop_around_count,on=['mall_id','shop_id'],how='left')
    del shop_around_count;gc.collect()

    count_feature['shop_around_ration_tfidf'] = count_feature['shop_around_count'] / (count_feature['mall_wifi_count'] + 1)

    count_feature['tfid_features'] = np.log1p(count_feature['shop_around_ration_tfidf']) * count_feature['wifi_shop_ratio_tfidf']

    # 构造集合特征

    count_feature = count_feature.fillna(0)
    return count_feature

# def get_current_feature(data):
#     data['current_week'] = pd.to_datetime(data['time_stamp']).map(lambda x:x.weekday())
#     # data['current_hour'] =
#     return data;

offline_train = train_wifi[(train_wifi['time_stamp'] < '2017-08-25 00:00')&(train_wifi['time_stamp'] >= '2017-08-18 00:00' )]
# print(offline_train.head())
offline_train_feature = train_wifi[(train_wifi['time_stamp'] < '2017-08-18 00:00')&(train_wifi['time_stamp'] >= '2017-08-04 00:00' )]
# print(offline_train_feature.head())
offline_train_feature = get_one_feature(offline_train_feature)
# print(offline_train_feature.head())


print(offline_train_feature.describe())
print(offline_train_feature.columns)
offline_train = make_features(offline_train_feature,offline_train)
# offline_train = get_current_feature(offline_train)
print(offline_train.head())
print(offline_train.columns)
offline_train.to_csv(dir + 'offline_train.csv',index=False)
del offline_train

offline_val = train_wifi[(train_wifi['time_stamp'] < '2017-09-01 00:00')&(train_wifi['time_stamp'] >= '2017-08-25 00:00' )]
# print(offline_val.head())
offline_val_feature = train_wifi[(train_wifi['time_stamp'] < '2017-08-25 00:00')&(train_wifi['time_stamp'] >= '2017-08-11 00:00' )]
# print(offline_val_feature.head())
offline_val_feature = get_one_feature(offline_val_feature)
# print(offline_val_feature.head())
offline_val = make_features(offline_val_feature,offline_val)
print(offline_val.columns)
offline_val.to_csv(dir + 'offline_val.csv',index=False)
del offline_val

sub_wifi = pd.read_csv(dir + 'test_wifi.csv')
# print(sub_wifi.head())
online_sub_feature = train_wifi[(train_wifi['time_stamp'] < '2017-09-01 00:00')&(train_wifi['time_stamp'] >= '2017-08-18 00:00' )]
# print(online_sub_feature.head())
online_sub_feature = get_one_feature(online_sub_feature)
# print(online_sub_feature.head())
sub_wifi = make_features(online_sub_feature,sub_wifi,row_id='row_id')
print(sub_wifi.columns)
sub_wifi.to_csv(dir + 'sub_wifi.csv',index=False)
del sub_wifi
