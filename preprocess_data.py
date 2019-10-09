#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:53:09 2019

@author: poganyg
"""

import pandas as pd


def remove_duplicate_IDs(df):

    duplicates = df['ID'].value_counts() > 1
    #print("Number of duplicates: ", sum(duplicates.astype(int)))

    #Extracting the ID-s of the duplicate entries
    duplicateIDs = duplicates.index[duplicates == True]

    #Drop the first value for each duplicates
    for duplicateID in duplicateIDs:
        indexes = df.ID[df.ID==duplicateID].index    
        df.drop(indexes[0], inplace=True)

    #print("Number of duplicates after dropping: ", sum((df['ID'].value_counts() > 1).astype(int)))
    
    return df


def make_sensible(df, remove_duplicates = True):
    
    sensible_df = pd.DataFrame()
    
    if remove_duplicates:
        df = remove_duplicate_IDs(df)
        print('Duplicate entries for the IDs removed')
    
    
    headers = ['ID','epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']
    
    unique_IDs = df.ID.str.slice(3,12).value_counts().index
    
    sensible_df[headers[0]] = unique_IDs
    
    for header in headers:
        if header != 'ID':            
            column_data = df['Label'][df['ID'].str.contains(header)].reset_index(drop=True)
            sensible_df[header] = column_data

    return sensible_df