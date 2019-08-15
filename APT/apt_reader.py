# -*- utf-8 -*-
"""
Created on 14 Aug, 2019.

Module to process apt data: store dataset into txt file, split dataset
into training dataset, validating dataset and test dataset.

@Author: Huang Zewen
"""

import re
import numpy as np
from pandas import DataFrame, read_csv
from apt_dbhelper import APT_MYSQL_TOKEN, APTDB
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_PATH = "./data/dataset.csv"
TRAINING_DATASET_PATH = "./data/train_dataset.csv"
VALIDATING_DATASET_PATH = "./data/validate_dataset.csv"
TESTING_DATASET_PATH = "./data/test_dataset.csv"


NUMBER_PATTERN = re.compile("\d+")
NUMER_REPLACE_TAG = ' '

def get_dataset(dataset_path=DATASET_PATH):
    apt_db = APTDB(*APT_MYSQL_TOKEN)
    log_tuple= apt_db.get_sh_err_logs_with_label()
    apt_db.close()
    log_list = []
    for log in log_tuple:
        log_list.append([re.sub(NUMBER_PATTERN, NUMER_REPLACE_TAG, str(log[0]).replace('\n', ' ')), log[1]])
    df = DataFrame(list(log_list), columns=['Log', 'Label'])
    df.to_csv(dataset_path, index=None, header=True)
    return df


def split_dataset(dataset, train_dataset_path=TRAINING_DATASET_PATH,
                  valid_dataset_path=VALIDATING_DATASET_PATH,
                  test_dataset_path=TESTING_DATASET_PATH):
    train, validate, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.8*len(dataset))])
    train.to_csv(train_dataset_path, index=None, header=True)
    validate.to_csv(valid_dataset_path, index=None, header=True)
    test.to_csv(test_dataset_path, index=None, header=True)


def describe_datasets(dataset_df, train_df, validate_df, test_df):
    fig, axes = plt.subplots(nrows=4, ncols=1)
    fig.subplots_adjust(hspace=0.5)
    for ax, df, name in zip(axes.flatten(), [dataset_df, train_df, validate_df, test_df],
                            ["Total Dataset", "Training Dataset", "Validate Dataset", "Test Dataset"]):
        sns.countplot(data=df, x='Label', ax=ax,
                      order=['apt_issue', 'ne_issue', 'case_cfg_env', 'instrument',
                             'ne_connection', 'traffic_verify_issue'])

        ax.set(title=name)
    fig.show()


if __name__ == "__main__":
    dataset_df = get_dataset()
    split_dataset(dataset_df)
    dataset_df = read_csv(DATASET_PATH)
    train_df = read_csv(TRAINING_DATASET_PATH)
    validate_df = read_csv(VALIDATING_DATASET_PATH)
    test_df = read_csv(TESTING_DATASET_PATH)
    # print(dataset_df)
    print(dataset_df.shape)
    print(train_df.shape)
    print(validate_df.shape)
    print(test_df.shape)
    print(dataset_df.info())
    describe_datasets(dataset_df, train_df, validate_df, test_df)
