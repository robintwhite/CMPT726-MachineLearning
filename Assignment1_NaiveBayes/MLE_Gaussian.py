# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 18:15:30 2017
    Maximum likelihood estimation for Gaussian distribution
    of given dataset of hockey player weight.
@author: Robin
"""

import pandas as pd
import numpy as np

#function for maximum likelihood of mu
def ML_mean(data):
    N = np.size(data)
    x = (1/N)*np.sum(data)
    return x

#function for maximum likelihood of sigma squared
def ML_std(data):
    N = np.size(data)
    s = (1/N)*np.sum((x - ML_mean(data))**2 for x in data)
    return s

#read data from file
df = pd.read_csv("normalized_datasets.csv")

#read single column data of player weight
weight = df["Weight_norm"].values
#read single column data of player weight conditional on GP > 0 being True
weight_gp_grt_0 = df.loc[df['GP_greater_than_0'] == 'yes', 'Weight_norm']
#read single column data of player weight conditional on GP > 0 being False
weight_gp_lst_0 = df.loc[df['GP_greater_than_0'] == 'no', 'Weight_norm']

##Write to text file for submission - see assignment submission for values
with open("Output.txt", "w") as text_file:
    
    print('{}'.format('Players weight, unconditional:'), file=text_file)
    print('{}{:.6f}'.format('Maximum likelihood estimate mu = ',ML_mean(weight)
    ), file=text_file)
    print('{}{:.6f}'.format('Maximum likelihood estimate sigma_squared = ',
          ML_std(weight)), file=text_file)
    print('', file=text_file)
    print('{}'.format('Players weight, conditional on GP > 0 == True:'),
          file=text_file)
    print('{}{:.6f}'.format('Maximum likelihood estimate mu = ',
          ML_mean(weight_gp_grt_0)), file=text_file)
    print('{}{:.6f}'.format('Maximum likelihood estimate sigma_squared = ',
          ML_std(weight_gp_grt_0)), file=text_file)
    print('', file=text_file)
    print('{}'.format('Players weight, conditional on GP > 0 == False:'),
          file=text_file)
    print('{}{:.6f}'.format('Maximum likelihood estimate mu = ',
          ML_mean(weight_gp_lst_0)), file=text_file)
    print('{}{:.6f}'.format('Maximum likelihood estimate sigma_squared = ',
          ML_std(weight_gp_lst_0)), file=text_file)
