# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:15:59 2017
    Using Naive Bayes to predict whether or not a player will play in the NHL 
    given a dataset of mixed variables, numbers and discrete catagories.
    The data is split into training and test data by the draft year.
    The data is also split by class to calculate the probabilities,
    this was done manually for simplicity since there are only two classes yes 
    and no
    The general outline is thus
    #go through each row, get probability for each variable for case GP_yes 
    and GP_no
    #log sum of probability ratios
    #if sum is greater than 0, then GP predicts yes, else no
    #count if determined correct outcome
@author: Robin White
"""


import pandas as pd
import numpy as np
import math


#function for maximum likelihood of mu
def ML_mean(data):
    N = np.size(data)
    x = (1/N)*np.sum(data)
    return x

#function for maximum likelihood of sigma squared
def ML_var(data):
    N = np.size(data)
    s = (1/N)*np.sum((x - ML_mean(data))**2 for x in data)
    return s

#function for gaussian distribution
def GProbability(x, mean, var):
	exponent = math.exp(-(math.pow(x - mean,2)/(2*var)))
	return (1 / (math.sqrt(2*math.pi*var))) * exponent

def GroupData(train_data):
    ##Summarize data into lookup table
    data_yes = train_data.loc[train_data['GP_greater_than_0'] == 'yes'].drop(
            "GP_greater_than_0", axis=1)
    data_no = train_data.loc[train_data['GP_greater_than_0'] == 'no'].drop(
            "GP_greater_than_0", axis=1)

    A = {}
    for column in data_yes:
        #create lookup table for probabilities for discrete values
        if data_yes[column].dtype == 'O':
            B = data_yes[column].value_counts()/data_yes[column].count()
            A[column] = pd.DataFrame(B, index=B.index)
            A[column].columns = ['yes']
        else:
            x = data_yes[column].values
            mean = ML_mean(x)
            var = ML_var(x)
            C = pd.Series([mean, var], index=['mean','var'])
            A[column] = pd.DataFrame(C, index=C.index)
            A[column].columns = ['yes']
    
    for column in data_no:
        #create lookup table for probabilities for discrete values
        if data_no[column].dtype == 'O':
            B = data_no[column].value_counts()/data_no[column].count()
            A[column]['no'] = pd.DataFrame(B, index=B.index)

        else:
            x = data_no[column].values
            mean = ML_mean(x)
            var = ML_var(x)
            C = pd.Series([mean, var], index=['mean','var'])
            A[column]['no'] = pd.DataFrame(C, index=C.index)
    return A

def getProbabilities(A, train_data, test_data):
    #Prior probabilities
    N = train_data['GP_greater_than_0'].count()
    P_yes = train_data[train_data["GP_greater_than_0"] == 'yes'].count(
            )["GP_greater_than_0"]/N
    P_no = train_data[train_data["GP_greater_than_0"] == 'no'].count(
            )["GP_greater_than_0"]/N
    
    #Seperate test into random variables and class
    df_test_x = test_data.drop("GP_greater_than_0", axis=1)
    df_test_y = test_data.loc[:,["GP_greater_than_0"]]
    
    pr_yes = np.zeros(len(df_test_x.columns))
    pr_no = np.zeros(len(df_test_x.columns))
    #iterate over all rows
    j = 0 #row count
    #empty string matrix
    getProbabilities.label = ["" for x in range(len(df_test_y))] 
    getProbabilities.P = np.zeros(len(df_test_y))
    count = 0

    for row in test_data.iterrows():
        #Note row[0] is index number, row[1] is data
        for i in range(len(df_test_x.columns)): #run through variables
            name = row[1].index[i] #column names for lookup
            value = row[1][i] #random variable value for test case
            if type(value) is str: #for discrete cases
                #conditional probability for value given yes
                pr_yes[i] = A[name].get_value(value,"yes")
                #conditional probability for value given no
                pr_no[i] = A[name].get_value(value,"no")
            else: #for continuous cases
                mean_y = A[name].get_value('mean',"yes")
                var_y = A[name].get_value('var',"yes")
                pr_yes[i] = GProbability(value, mean_y, var_y)
                mean_n = A[name].get_value('mean',"no")
                var_n = A[name].get_value('var',"no")
                pr_no[i] = GProbability(value, mean_n, var_n)

        #log sum of odds
        for i in range(len(df_test_x.columns)):
            getProbabilities.P[j] += math.log(pr_yes[i]/pr_no[i])
        #include priors
        getProbabilities.P[j] = getProbabilities.P[j] + math.log(P_yes/P_no) 
        
        if getProbabilities.P[j] >= 0:
            getProbabilities.label[j] = 'yes'
        else:
            getProbabilities.label[j] = 'no'
        #Check if prediction is correct
        if getProbabilities.label[j] == df_test_y.get_value(row[0]
        , 'GP_greater_than_0'):
            count += 1
        
        #row count since index does not increment 
        #monotomically from splitting of data
        j += 1
        
    return count/len(df_test_x)
                 
 
#read data from file
df = pd.read_csv("normalized_datasets.csv")
#Seperate into training and test set
#Training from yrs 1998,1999,2000
df_train = df.loc[(df['DraftYear'] == 1998) |
                  (df['DraftYear'] == 1999) |
                  (df['DraftYear'] == 2000)]
#Test from yr 2001
df_test = df.loc[(df['DraftYear'] == 2001)]
#Drop columns as given on course website, returns new dataset
df_train = df_train.drop(["id","PlayerName","DraftYear","po_PlusMinus_norm",
                          "sum_7yr_GP","sum_7yr_TOI"], axis=1)
df_test = df_test.drop(["id","PlayerName","DraftYear","po_PlusMinus_norm",
                        "sum_7yr_GP","sum_7yr_TOI"], axis=1)
       
data_grouped = GroupData(df_train)
ratio = getProbabilities(data_grouped, df_train, df_test)
labels = getProbabilities.label #List of predicted class labels
log_sum = getProbabilities.P #List of log_sum for each test instance 

print('{}{:.6f}{}'.format('Accuracy of prediction: ', ratio*100, '%'))
import csv
f = open('grouped_data_vals.csv', 'w')
w = csv.writer(f, delimiter=',', quotechar='|')
for key, val in data_grouped.items():
    w.writerow([key, val])
f.close()
##Output: Accuracy of prediction: 70.081967%