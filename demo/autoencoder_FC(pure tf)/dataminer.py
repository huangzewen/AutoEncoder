# -*- coding: utf-8 -*-

"""
Created on 22 Jul, 2019.

Module to process data.

@author: Huang Zewen
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams

from numpy.random import seed

DATA_SPLIT_PCT = 0.2

df = pd.read_csv("data/processminer-rare-event-mts - data.csv")

print(df.head())

sign = lambda x: (1, -1)[x < 0]

SEED = 123

def curve_shift(df, shift_by):
    '''
    This function will shift the binary labels in a dataframe.
    The curve shift will be with respect to the 1s.
    For example, if shift is -2, the following process
    will happen: if row n is labeled as 1, then
    - Make row (n+shift_by):(n+shift_by-1) = 1.
    - Remove row n.
    i.e. the labels will be shifted up to 2 rows up.

    Inputs:
    df       A pandas dataframe with a binary labeled column.
             This labeled column should be named as 'y'.
    shift_by An integer denoting the number of rows to shift.

    Output
    df       A dataframe with the binary labels shifted by shift.
    '''

    vector = df['y'].copy()
    for s in range(abs(shift_by)):
        tmp = vector.shift(sign(shift_by))
        tmp = tmp.fillna(0)
        vector += tmp
    labelcol = 'y'
    # Add vector to the df
    df.insert(loc=0, column=labelcol+'tmp', value=vector)
    # Remove the rows with labelcol == 1.
    df = df.drop(df[df[labelcol] == 1].index)
    # Drop labelcol and rename the tmp col as labelcol
    df = df.drop(labelcol, axis=1)
    df = df.rename(columns={labelcol+'tmp': labelcol})
    # Make the labelcol binary
    df.loc[df[labelcol] > 0, labelcol] = 1

    return df


df = curve_shift(df, -2)
df = df.drop(["time", "x28", "x61"], axis=1)

df_train, df_test = train_test_split(df, test_size=DATA_SPLIT_PCT,
                                     random_state=SEED)

df_train, df_valid = train_test_split(df_train, test_size=DATA_SPLIT_PCT,
                                      random_state=SEED)

df_train_0 = df_train.loc[df['y'] == 0]
df_train_1 = df_train.loc[df['y'] == 1]
df_train_0_x = df_train_0.drop(['y'], axis=1)
df_train_1_x = df_train_1.drop(['y'], axis=1)

df_valid_0 = df_valid.loc[df["y"] == 0]
df_valid_1 = df_valid.loc[df["y"] == 1]
df_valid_0_x = df_valid_0.drop(['y'], axis=1)
df_valid_1_x = df_valid_1.drop(['y'], axis=1)

df_test_0 = df_test.loc[df["y"] == 0]
df_test_1 = df_test.loc[df["y"] == 1]
df_test_0_x = df_test_0.drop(["y"], axis=1)
df_test_1_x = df_test_1.drop(["y"], axis=1)

scaler = StandardScaler().fit(df_train_0_x)
df_train_0_x_rescaled = scaler.transform(df_train_0_x)

df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)
df_valid_1_x_rescaled = scaler.transform(df_valid_1_x)

df_valid_x_rescaled = scaler.transform(df_valid.drop(['y'], axis=1))

df_test_0_x_rescaled = scaler.transform(df_test_0_x)
df_test_x_rescaled = scaler.transform(df_test.drop(['y'], axis=1))

def get_train_0_x_rescaled():
    return df_train_0_x_rescaled

def get_valid_0_x_rescaled():
    return df_valid_0_x_rescaled

def get_valid_1_x_rescaled():
    return df_valid_1_x_rescaled

def get_test_0_rescaled():
    return df_test_0_x_rescaled


if __name__ == "__main__":
    df_train = get_train_0_x_rescaled()
    df_train = pd.DataFrame(df_train)
    print(df_train)
    print(len(df_train))
    x_batches = np.array_split(df_train, 500)
    print(len(x_batches))
    print(len(x_batches[-1]))
    print(len(x_batches[-2]))





