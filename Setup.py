#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:15:11 2019

@author: docear
"""

' This code is adapted from: https://github.com/radekosmulski/rsna-intracranial/blob/master/02_reshape_train_csv.ipynb '

import pandas as pd
revisedtrainCSVpath = '/media/docear/My Passport/Kaggle/Hemorrhage/stage_1_train_revised.csv'

df_train = pd.read_csv(revisedtrainCSVpath)

df_train['fn'] = df_train.ID.apply(lambda x: '_'.join(x.split('_')[:2]) + '.png')
df_train.columns = ['ID', 'probability', 'fn']
df_train['label'] = df_train.ID.apply(lambda x: x.split('_')[-1])

df_train.drop_duplicates('ID', inplace=True)
pivot = df_train.pivot(index='fn', columns='label', values='probability')
pivot.head()

pivot.reset_index(inplace=True)
pivot.head()

from collections import defaultdict
d = defaultdict(list)
for fn in df_train.fn.unique(): d[fn]
for tup in df_train.itertuples():
    if tup.probability: d[tup.fn].append(tup.label)
    
ks, vs = [], []

for k, v in d.items():
    ks.append(k), vs.append(' '.join(v))
    
pd.DataFrame(data={'fn': ks, 'labels': vs}).to_csv('/media/docear/My Passport/Kaggle/Hemorrhage/train_labels_as_strings.csv', index=False)
