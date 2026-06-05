import pandas as pd 
import torch
import numpy as np

def get_smoothed_midprice_targets(df,k = 100):
    '''
    gets the smoothed midprice targets for the given dataframe
    and returns only the required columns (removing the message columns)
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
    #normalize X_train per feature
    X_train_mean = X_train.mean(dim=0)
    X_train_std = X_train.std(dim=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    return X_train,X_val,X_test

def make_target_labels(y_train,y_val,y_test):
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