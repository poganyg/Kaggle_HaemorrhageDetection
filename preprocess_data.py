#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:53:09 2019

@author: poganyg
"""

import pandas as pd


def _remove_duplicate_IDs(df):

    print("Removing duplicate entries..")
    duplicates = df['ID'].value_counts() > 1
    #print("Number of duplicates: ", sum(duplicates.astype(int)))

    #Extracting the ID-s of the duplicate entries
    duplicateIDs = duplicates.index[duplicates == True]

    #Drop the first value for each duplicates
    for duplicateID in duplicateIDs:
        indexes = df.ID[df.ID==duplicateID].index    
        df.drop(indexes[0], inplace=True)
    
    print("Removed duplicate entries.")
    return df


def make_sensible(df, remove_duplicates = True, writeout = False):
    """
    df: pandas dataframe
    
    remove_duplicates: unless duplicates have been removed prior to function
    call, leave as True
    
    writeout: if new dataframe is to be saved, pass a string with the desired
    filename, e.g. 'test.csv'
    """
    
    sensible_df = pd.DataFrame()
    
    if remove_duplicates:
        df = _remove_duplicate_IDs(df)
    
    
    headers = ['ID','epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']
    
    unique_IDs = df.ID.str.slice(3,12).value_counts().index
    
    sensible_df[headers[0]] = unique_IDs
    
    print("Constructing the new dataframe..")
    for header in headers:
        if header != 'ID':            
            column_data = df['Label'][df['ID'].str.contains(header)].reset_index(drop=True)
            sensible_df[header] = column_data
    print("Dataframe reformated.")
            
    if writeout:
        print("Saving dataframe as: "+writeout+"..")
        sensible_df.to_csv(writeout,index=False)
        print("Dataframe saved as: "+writeout)

    return sensible_df