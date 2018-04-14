# From: https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing

import pandas as pd
import numpy as np

df = pd.read_csv('filtered_labels.csv')

groups = df.groupby('filename')                                                                                                                                                                                                                                                                    
keys = list(groups.groups.keys())                                                                                                                                                                                                                                                                  
keys = pd.Series(keys)                                                                                                                                                                                                                                                                             
msk = np.random.rand(len(keys)) <= 0.8                                                                                                                                                                                                                                                             
train_keys = keys[msk]                                                                                                                                                                                                                                                                             
test_keys = keys[~msk]                                                                                                                                                                                                                                                                             
train_keys = list(train_keys.values)                                                                                                                                                                                                                                                               
msk = df['filename'].apply(lambda x: x in train_keys)                                                                                                                                                                                                                                             


train = df[msk]
test = df[~msk]

train.to_csv('train_labels.csv', index=False)
test.to_csv('test_labels.csv', index=False)
