# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:45:28 2017
    Pre-processing of data before training
@author: Robin
"""

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

#Could use numpy, but required to write ourselves
#function for mu (mean)
def ML_mean(data):
    N = np.size(data)
    x = (1/N)*np.sum(data)
    return x

#function for sigma squared (variance)
def ML_var(data):
    N = np.size(data)
    s = (1/N)*np.sum((x - ML_mean(data))**2 for x in data)
    return s

def standardize(ndata):
    #standardize continuous data and create boolean variables for discrete
    #variables
    data = copy.deepcopy(ndata)
    s = '_'
    j = 0
    tmp = np.zeros(len(data))
    for column in data:
        #create unqie variable for discrete catagories
        if data[column].dtype == 'O':
            a = data[column].unique()
            for i in range(a.size):
                k = 0
                for row in data.iterrows():                
                    if row[1][column] == a[i]:
                        tmp[k] = int(1)
                    else:
                        tmp[k] = int(0)
                    k += 1 #row index
                data.insert(j+1, s.join((column,a[i])), tmp)
            data = data.drop([column], axis=1)   
        else:
            x = data[column].values
            mean = ML_mean(x)
            var = ML_var(x)
            data[column] = (data[column]-mean)/var
            
        j += 1 #column index
     
    return data

def interactions(ndata):
    data_old = copy.deepcopy(ndata)
    data = copy.deepcopy(ndata)
    s = '_'
    j = data.shape[1] - 1
    for column_i in data_old:
        x = data[column_i]
        for column_j in data_old:
            if column_j != column_i and s.join((column_j,column_i)) not in data.columns:
                y = data[column_j]
                inter_terms = np.multiply(x,y)
                data.insert(j+1, s.join((column_i,column_j)), inter_terms)
                j += 1 #column index

    return data

#compute error
#Mean sqared error with 1/2N factor infront
#Better for comparison against test data with smaller sample size
def squaredErr(weightV, x_data, t_data):
    w_matrix = np.matrix(weightV)
    x_matrix = np.matrix(x_data)
    t_matrix = np.matrix(t_data)
    cost = (t_matrix.T - x_matrix*w_matrix)
    return cost.T*cost

#compute weights
def normalEq(L, x_data, t_data):
    I = np.identity(x_data.shape[1])
    x_matrix = np.matrix(x_data)
    t_matrix = np.matrix(t_data)
    return np.linalg.pinv(L*I - x_matrix.T*x_matrix)*x_matrix.T*t_matrix.T

#read data from file
df = pd.read_csv(r"preprocessed_datasets.csv")

class_name = 'sum_7yr_GP'

##Seperate into training and test set
#Training from yrs 1998,1999,2000
df_train = df.loc[(df['DraftYear'] == 2004) |
                  (df['DraftYear'] == 2005) |
                  (df['DraftYear'] == 2006)]
#Test from yr 2001
df_test = df.loc[(df['DraftYear'] == 2007)]

list_of_dropped_vars = ["id","PlayerName","DraftYear","Country",
                          "Overall","sum_7yr_TOI","GP_greater_than_0"]

#Drop columns as given on course website, returns new dataset
df_train = df_train.drop(list_of_dropped_vars, axis=1)
df_test = df_test.drop(list_of_dropped_vars, axis=1)

#%% Pre process data
x_train = standardize(df_train)
x_test = standardize(df_test)

#Target classes
t_train = x_train[class_name]
t_test = x_test[class_name]

#Training and test Data
x_train = x_train.drop([class_name], axis=1)
x_test = x_test.drop([class_name], axis=1)

#add interaction terms for all i, j: xi*xj
x_train = interactions(x_train)
x_test = interactions(x_test)

#Insert w0 term for weight vector matrix
x_train.insert(0, 'w0', np.ones(len(x_train), dtype=np.int))
x_test.insert(0, 'w0', np.ones(len(x_test), dtype=np.int))

#%% Cross validation 
#k-fold cross validation
#using 10 in this case
N = 10 #number of divisions 
lambdas =[0,0.01,0.1,1,10,100,1000]

#shuffle dataset 
temp_data = x_train.join(t_train)
temp_data = temp_data.reindex(np.random.permutation(temp_data.index))
temp_data = temp_data.reset_index(drop=True) #reset index to call rows

#Cross validation set-up
if N <= 2:
    div = len(temp_data)//(N) #size of groups for N-fold cross validation
    LossList = np.zeros(N+1)
else:
    div = len(temp_data)//(N-1)
    LossList = np.zeros(N)
    
weightVavg = [] #Weight list for each lambda
validationError = np.zeros(len(lambdas)) #average loss list
testError = np.zeros(len(lambdas)) #test loss list

#Make groups of data
grouped = temp_data.groupby(temp_data.index.to_series() // div)
#iterate through each group, assign as test set and rest as train set, 
#move through each group
i = 0
for L in lambdas:
    weightVList = [] #Weight list for validation runs
    #validation
    for group_id, group in grouped: 
        #test group
        t_group = grouped.get_group(group_id) 
        t_dataTarget = t_group[class_name] 
        t_data = t_group.drop([class_name], axis=1)
        #train group
        x_group = temp_data.drop(t_group.index.to_series())
        x_dataTarget = x_group[class_name]
        x_data = x_group.drop([class_name], axis=1)
        
        #weightV = normalEq(L, x_data, x_dataTarget)
        #Note .append inserts to 0 position
        #List of weights for trained data on each validation set
        weightVList.insert(group_id, normalEq(L, x_data, x_dataTarget).tolist())
        #Squared loss for test data in validation set Ignore:(1/(2*np.sqrt(ML_var(t_dataTarget))*len(t_data)))*
        LossList[group_id] = squaredErr(weightVList[group_id], t_data, t_dataTarget)
        
    validationError[i] = ML_mean(LossList)
    a = np.asarray(weightVList)
    #List of average weight values for each lambda
    weightVavg.insert(i, [ML_mean(x) for x in np.asmatrix(a.reshape(a.shape[0],a.shape[1])).T])
    #Insert at end of array
    
    #test Ignore:(1/(2*np.sqrt(ML_var(t_test))*len(x_test)))*
    t = np.asarray(weightVavg)[i]
    testError[i] = squaredErr(np.transpose(np.asmatrix(t)), x_test, t_test)
    
    i += 1


#%%
#Results
lmb = lambdas[validationError.tolist().index(min(validationError))]
error = validationError[validationError.tolist().index(min(validationError))]
test_error = testError[validationError.tolist().index(min(validationError))]
print("{}{:.4f}".format("Best Lambda: ", lmb))
print("{}{:.6f}".format("Validation Error at Best Lambda: ", error))
print("{}{:.6f}".format("Test Error at Best Lambda: ", test_error))
#%%Plot
fig, ax1 = plt.subplots()
lns1 = ax1.plot(lambdas, validationError, 'b-',label='Validation error')
lns2 = ax1.plot(lmb,error,marker='o',color='k',label="Best Lambda")
ax1.set_xscale('symlog')
ax1.set_ylabel('Sum Squared Error', color='k')
ax1.tick_params('y', colors='k')
ax1.set_xlabel('Lambda')

#ax2 = ax1.twinx() #For secondary y-axis
lns3 = ax1.plot(lambdas, testError,'r-', label='Test error')
#ax2.set_ylabel('Sum Squared Error', color='r')
#ax2.tick_params('y', colors='r')

#ax.semilogx(lambdas, validationError,label='Validation error')
#ax.semilogx(lambdas, testError,label='test error')
#ax.semilogx(lmb,error,marker='o',color='r',label="Best Lambda")

# Legend
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

fig.tight_layout()
plt.show()