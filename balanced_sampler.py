# -*- coding: utf-8 -*-

import pandas as pd
import random

path = '/media/poganyg/Kaggle/RSNA/rsna-intracranial-hemorrhage-detection/'

labels = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']

def random_sample_n(n, filename):
    """
    n - how many samples you want
    filename - the csv file from which you want the samples from (with columns
    'ID' and 'Label')
    
    It's probably not a good idea to make n < 13 because the function samples 1 
    positive and 1 negative example from all 6 label-classes
    
    For now it won't necessarily give you exactly n samples (only if n is exactly
    divisible by 12), it might be up to 5 samples fewer returned, should be fixed
    """
    df = pd.read_csv(path+filename)
    frames = []
    n = int(n/2 // 6)
    
    for label in labels: 
        
        subset = df[(df.ID.str.contains(label)) & (df.Label == 1)]
        asd = sorted(random.sample(range(len(subset)),n))
        sampledIDs = subset['ID'].iloc[asd]
        
        frames.append(sampledIDs)
        
        subset = df[(df.ID.str.contains(label)) & (df.Label == 0)]
        asd = sorted(random.sample(range(len(subset)),n))
        sampledIDs = subset['ID'].iloc[asd]
        
        frames.append(sampledIDs)
        
    
    return(pd.concat(frames).sample(frac=1))    