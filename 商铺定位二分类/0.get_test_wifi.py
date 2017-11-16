#coding:utf-8
# 引入需要的包

import pandas as pd
from tqdm import tqdm

'''
读取数据
'''
dire = './data/'
print('test_wifi_set_')
test_behavior = pd.read_csv(dire + 'evaluation_public.csv')

# 提取wifi信息，分割为 bassid wifi_strength 格式
def get_wifi_dict(df):
    '''
    :param df: 待处理数据
    :return: 将每条记录的所有bssid分开成多个行，得到处理好的WiFi信息并返回
    '''
    # 构造WiFi字典
    wifiDict={
        # 注意训练集和测试集的区别，训练集需要添加shop_id，测试集不需要
        'row_id': [],
        'bssid': [],
        'strength': [],
        'connect': [],
        'mall_id': [],
        'nature_order':[]

     }
    for index, row in tqdm(df.iterrows()):
        # print(index)
        order_index = 1
        for wifi in row.wifi_infos.split(';'):
            info = wifi.split('|')
            wifiDict['row_id'].append(row.row_id)
            wifiDict['mall_id'].append(row.mall_id)
            wifiDict['bssid'].append(info[0])
            wifiDict['strength'].append(info[1])
            wifiDict['connect'].append(info[2])
            wifiDict['nature_order'].append(order_index)
            order_index = order_index + 1

    print('done')
    del df
    wifi = pd.DataFrame(wifiDict)
    return wifi

print(test_behavior.shape)
test_wifi = get_wifi_dict(test_behavior)
print(test_wifi.shape)
test_wifi.to_csv(dire + 'test_wifi.csv', index=False)