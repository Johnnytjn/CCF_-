#coding:utf-8
import pandas as pd
dir = './data/'
sub = pd.read_csv(dir + 'evaluation_public.csv')
print(sub.shape)
sub_1 = pd.read_csv('./tmp.csv')

sub = pd.merge(sub,sub_1,on=['row_id'],how='left')

sub = sub[['row_id','shop_id']]
sub = sub.fillna('s_167275')

sub[['row_id','shop_id']].to_csv('./result.csv',index=None)
print(sub)
