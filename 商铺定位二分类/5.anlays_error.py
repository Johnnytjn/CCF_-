import pandas as pd


anlayse = pd.read_csv('./data/offline_train.csv')
# anlayse['true'] = (anlayse['shop_id']==anlayse['shop_id_y'])

print(anlayse.groupby(['label']).describe().reset_index().to_csv('./1.csv',index=False))