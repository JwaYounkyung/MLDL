import pandas as pd 
from functools import reduce

def dataPreprocessX(dataset):

    #activity data
    df_activity = pd.read_csv('./raw/' + dataset + '_activity.csv',header=0)
    df_activity_gp = df_activity.groupby(['day','acc_id'],as_index=False).sum()
    df_activity_gp = df_activity_gp.groupby(['acc_id'],as_index=False).mean()
    df_activity_gp.drop(['char_id', 'day'], axis='columns', inplace=True)
    
    #combat data
    df_combat = pd.read_csv('./raw/' + dataset + '_combat.csv',header=0)
    df_combat_gp = df_combat.groupby(['day','acc_id'],as_index=False).sum()
    df_combat_gp = df_combat_gp.groupby(['acc_id'],as_index=False).mean()
    df_combat_gp.drop(['char_id','day'], axis='columns', inplace=True) #class 살릴까 말까

    #payment data
    df_payment = pd.read_csv('./raw/' + dataset + '_payment.csv',header=0)
    df_payment_gp = df_payment.groupby(['acc_id'],as_index=False).mean()
    df_payment_gp.drop(['day'], axis='columns', inplace=True)

    #pledge data
    df_pledge = pd.read_csv('./raw/' + dataset + '_pledge.csv',header=0)
    df_pledge_gp = df_pledge.groupby(['day','acc_id'],as_index=False).sum()
    df_pledge_gp = df_pledge.groupby(['acc_id'],as_index=False).mean()
    df_pledge_gp.drop(['char_id', 'pledge_id', 'day'], axis='columns', inplace=True)

    #trade data
    df_trade = pd.read_csv('./raw/' + dataset + '_trade.csv',header=0)
    df_trade_source = df_trade.iloc[:, [0,4,9,10]].fillna(0) #process NaN
    df_trade_target = df_trade.iloc[:, [0,6,9,10]].fillna(0)

    df_trade_source_gp = df_trade_source.groupby(['day','source_acc_id'],as_index=False).sum()
    df_trade_source_gp = df_trade_source_gp.groupby(['source_acc_id'],as_index=False).mean()
    df_trade_source_gp.rename(columns = {"source_acc_id": "acc_id", "item_amount":"source_item_amount", "item_price":"source_item_price"}, inplace = True)
    df_trade_source_gp.drop(['day'], axis='columns', inplace=True)

    df_trade_target_gp = df_trade_target.groupby(['day','target_acc_id'],as_index=False).sum()
    df_trade_target_gp = df_trade_target_gp.groupby(['target_acc_id'],as_index=False).mean()
    df_trade_target_gp.rename(columns = {"target_acc_id": "acc_id", "item_amount":"target_item_amount", "item_price":"target_item_price"}, inplace = True)
    df_trade_target_gp.drop(['day'], axis='columns', inplace=True)

    #data merging
    dfs1 = [df_activity_gp, df_combat_gp, df_payment_gp, df_pledge_gp]
    df_merge_1 = reduce(lambda left,right: pd.merge(left,right,on='acc_id', how='outer'), dfs1)
    df_merge_1 = df_merge_1.fillna(0)

    dfs2 = [df_merge_1, df_trade_source_gp, df_trade_target_gp]
    df_merge_2 = reduce(lambda left,right: pd.merge(left,right,on='acc_id', how='left'), dfs2)
    df_x = df_merge_2.fillna(0)
    
    series_x_id = df_x['acc_id']
    frame = { 'acc_id': series_x_id } 
    df_x_id = pd.DataFrame(frame) 
    
    del df_x['acc_id'] #delete acc_id 

    #save to csv
    df_x.to_csv('./preprocess/' + dataset + '_preprocess_1.csv', index=False)
    df_x_id.to_csv('./preprocess/' + dataset + '_id_1.csv', index=False)
    print(dataset + 'sucess')

def dataPreprocessY(dataset):

    #sorting by acc_id
    df_label = pd.read_csv('./raw/' + dataset + '_label.csv',header=0)
    df_label = df_label.sort_values(by=['acc_id'], axis=0)

    #save to csv
    df_label.to_csv('./preprocess/' + dataset + '_label_1.csv', index=False)
    print(dataset + 'sucess')

dataPreprocessX('train')
dataPreprocessX('test1')
dataPreprocessX('test2')
dataPreprocessY('train')