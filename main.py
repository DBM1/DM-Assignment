import numpy as np
import pandas as pd
import sklearn

CATEGORY = 'CATEGORY'
TEXT = 'TEXT'


def load_data(nrows=None) -> pd.DataFrame:
    data_file = 'tweet/training.1600000.processed.noemoticon.csv'
    columns_index = [0, 5]
    columns_name = [CATEGORY, TEXT]
    file_data = pd.read_csv(data_file, engine='python', header=None, usecols=columns_index, names=columns_name,
                            nrows=nrows)
    file_data = file_data.loc[:, [TEXT, CATEGORY]]
    return file_data


data = load_data()
print(data.loc[:, CATEGORY].value_counts())
