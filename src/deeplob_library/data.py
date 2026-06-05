import pandas as pd 
import torch
import numpy as np

def get_smoothed_midprice_targets(df,k = 100):
    '''
    df: dataframe of message and orderbook
    k: lookback window
    returns: dataframe of smoothed midprice targets
    '''
    df = df.iloc[:,6:]
    #get midprice
    df['midprice'] = (df['bid_price_1'] + df['ask_price_1'])/2
    #get smoothed midprice
    df['midprice_smooth'] = df['midprice'].rolling(window=k).mean()
    #get targets
    df['target'] = df['midprice_smooth'].shift(-k)/df['midprice_smooth'] - 1

    return df.dropna().drop(columns=['midprice','midprice_smooth'])

def normalize_train_val_test(X_train,X_val,X_test):
    '''
    X_train: tensor 
    X_val: tensor 
    X_test: tensor 
    returns: tuple of tensors of X_train, X_val, X_test normalized per feature
    '''
    #normalize X_train per feature
    X_train_mean = X_train.mean(dim=0)
    X_train_std = X_train.std(dim=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    return X_train,X_val,X_test

def make_target_labels(y_train,y_val,y_test):
    '''
    y_train: tensor of shape (batch_size,)
    y_val: tensor of shape (batch_size,)
    y_test: tensor of shape (batch_size,)
    returns: tuple of tensors of y_train, y_val, y_test as balanced ternary labels
    '''
    #make into balanced ternary labels

    #get thresholds
    low_threshold = torch.quantile(y_train,0.33)
    high_threshold = torch.quantile(y_train,0.66)

    y_train = torch.where(y_train < low_threshold,0,
                          torch.where(y_train > high_threshold,2,1))
    y_val = torch.where(y_val < low_threshold,0,
                        torch.where(y_val > high_threshold,2,1))
    y_test = torch.where(y_test < low_threshold,0,
                         torch.where(y_test > high_threshold,2,1))
    return y_train.long(),y_val.long(),y_test.long()

def read_message_file(file_path):
    '''
    file_path: path to message file
    returns: dataframe of message file
    '''
    message_cols = ["time","event_type","order_id","size","price","direction","extra"]
    df = pd.read_csv(file_path,names=message_cols)
    #drop extra column
    df = df.drop(columns=["extra"])
    return df

def read_orderbook_file(file_path):
    '''
    file_path: path to orderbook file
    returns: dataframe of orderbook file
    '''
    lab_cols = []
    for i in range(1,11):

        lab_cols.append(f"ask_price_{i}")
        lab_cols.append(f"ask_size_{i}")

        lab_cols.append(f"bid_price_{i}")
        lab_cols.append(f"bid_size_{i}")
    df = pd.read_csv(file_path,names=lab_cols)
    return df


def combine_message_orderbook(message_df,orderbook_df):
    '''
    message_df: dataframe of message file
    orderbook_df: dataframe of orderbook file

    returns: dataframe of combined message and orderbook
    '''

    #concat
    df = pd.concat([message_df,orderbook_df],axis=1)
    df_regular = df[
        (df['time'] > 10*60*60) &
        (df['time'] < 15.5*60*60) &
        (df['event_type'].isin([1,2,3,4,5]))
    ]
    return df_regular

#get orderbook for each day and concat into tensors of 100 lookback
def get_lists_of_tensors(sorted_message_files:list[str],
                        sorted_orderbook_files:list[str],
                        ) -> tuple[list[torch.Tensor],list[torch.Tensor]]:

    '''
    sorted_message_files: list of message files sorted by date
    sorted_orderbook_files: list of orderbook files sorted by date

    returns: tuple of lists of tensors of X and y

    X: tensor of shape (batch_size, 100, 10)
    y: tensor of shape (batch_size,)
    where 100 is the lookback window and 10 is the number of features
    '''
    #get orderbook for each day
    tensors_X = []
    tensors_y = []
    for message_file,orderbook_file in zip(sorted_message_files,sorted_orderbook_files):

        message_df = read_message_file(message_file)
        orderbook_df = read_orderbook_file(orderbook_file)
        df = combine_message_orderbook(message_df,orderbook_df)
        df = get_smoothed_midprice_targets(df)
        
        X = torch.tensor(df.iloc[:,:-1].values).float()
        y = torch.tensor(df.iloc[:,-1].values).float()
        X = X.unfold(0,100,10).permute(0,2,1)
        y = y.unfold(0,100,10)[:,-1]
        tensors_X.append(X)
        tensors_y.append(y)

        
    return tensors_X,tensors_y

