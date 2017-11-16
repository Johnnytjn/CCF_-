#coding:utf-8
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from tools import get_features_list,hafuman_km,bearing_array
# load model to predict

features = get_features_list()

dir = './data/'
shop_info_tmp = pd.read_csv(dir + 'ccf_first_round_shop_info.csv')
shop_info = shop_info_tmp[['shop_id','category_id','longitude','latitude','price']]
del shop_info_tmp;gc.collect()
shop_info.rename(columns={'longitude':'s_longitude','latitude':'s_latitude'},inplace=True)

print('Load model to predict')

gbm = lgb.Booster(model_file='model.txt')

# 提交数据
sub = pd.read_csv(dir + 'sub_wifi.csv')
print(sub.shape)
sub_user_info = pd.read_csv(dir + 'evaluation_public.csv')
# row_id,user_id,mall_id,time_stamp,longitude,latitude,wifi_infos
sub_user_info = sub_user_info[['row_id','longitude','latitude','time_stamp']]
sub = pd.merge(sub,shop_info,on=['shop_id'],how='left')
del sub['label']
del shop_info;gc.collect()
# 暴力删除1万个nan数据
sub = sub.dropna()
sub = pd.merge(sub,sub_user_info,on=['row_id'],how='left')
print(sub.head)
sub['time_stamp'] = pd.to_datetime(sub['time_stamp'])
sub['current_hour'] =  pd.DatetimeIndex(sub.time_stamp).hour
# sub['current_week'] =  pd.DatetimeIndex(sub.time_stamp).dayofweek

sub['distance'] = hafuman_km(sub['s_longitude'],sub['s_latitude'],sub['longitude'],sub['latitude'])
sub['current_bearing_array'] = bearing_array(sub.s_latitude.values, sub.s_longitude.values, 
                                                                                    sub.latitude.values, sub.longitude.values)  
                                                                                    
#sub['distance'] = np.log1p(sub['distance'])
sub['c_wifi_var'] = sub['strength'] - sub['c_sw_average']
sub['wifi_var'] = sub['strength'] - sub['s_avg_power']
sub['s_wifi_var'] = sub['strength'] - sub['w_avg_power']
sub['sb_wifi_var'] = sub['strength'] - sub['sb_history_avg_power']
sub['c_sb_wifi_var'] = sub['strength'] - sub['c_sb_history_avg_power']

sub['angle_var'] = sub['history_bearing_array_median'] - sub['current_bearing_array']

sub['s_sb_wifi_var_ratio'] = sub['c_sb_wifi_var'] / (sub['sb_wifi_var']+0.0001)

sub['category_id'] = sub['category_id'].map(lambda x:str(x).split('_')[1])
sub['mall_id'] = sub['mall_id'].map(lambda x:str(x).split('_')[1])
sub['shop_id_f'] = sub['shop_id'].map(lambda x:str(x).split('_')[1])
sub['bssid_f'] = sub['bssid'].map(lambda x:str(x).split('_')[1])
print(sub.head())
sub_r_s = sub[['row_id','shop_id']]
sub_ = sub[features]
del sub;gc.collect()
print(sub_.columns)

sub_lgb = sub_[features].values

result = gbm.predict(sub_lgb)
result = pd.DataFrame({'pre_p':list(result)})
result = pd.concat([sub_r_s,result],axis=1)
del sub_;gc.collect()
del sub_r_s;gc.collect()
result = pd.DataFrame(result).sort_values('pre_p',ascending=False).drop_duplicates('row_id')
print(result.shape)
result[['row_id','shop_id','pre_p']].to_csv('./tmp.csv',index=None)
