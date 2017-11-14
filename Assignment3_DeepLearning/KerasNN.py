# -*- coding: utf-8 -*-
"""
    Implementation of deep learning using Keras and Tensorflow
    to predict number of games played after 7 years in the NHL
    from given dataset
    Using some preproccesing code implemented in Assignment 2
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import regularizers
from keras import backend
from keras import metrics

#data handling
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy
from time import time

# Visualisation
import matplotlib.pyplot as plt


#%% Standardize values to 0 and 1
def standardize(ndata, b):
    #standardize continuous data and create boolean variables for discrete
    #variables
    data = copy.deepcopy(ndata)

    for column in data:
        #create unqie variable for discrete catagories
        if any(column != t for t in b):
            x = data[column].values
            x = x.reshape((len(x), 1))
            scaler = StandardScaler()
            scaler = scaler.fit(x)
            data[column] = scaler.transform(x)
            
     
    return data

def dummy(ndata):
    #create boolean variables for discrete variables
    data = copy.deepcopy(ndata)
    s = '_'
    b = []
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
                b.append(s.join((column,a[i])))
            data = data.drop([column], axis=1)   
            
        j += 1 #column index
     
    return data, b

#%%    
def squaredErr(y_true, y_pred):
    return backend.sum(backend.square(y_pred - y_true))

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

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
#%% Load in data
#read data from file
df = pd.read_csv(r"preprocessed_datasets.csv")

#shuffle dataset 
np.random.seed(7)
temp_data = df.reindex(np.random.permutation(df.index))
temp_data = temp_data.reset_index(drop=True) #reset index to call rows

class_name = 'sum_7yr_GP'

##Seperate into training and test set
#Training from yrs 1998,1999,2000
df_train = temp_data.loc[(df['DraftYear'] == 2004) |
                  (df['DraftYear'] == 2005) |
                  (df['DraftYear'] == 2006)]
#Test from yr 2001
df_test = temp_data.loc[(df['DraftYear'] == 2007)]

list_of_dropped_vars = ["id","PlayerName","DraftYear","Country",
                          "Overall","sum_7yr_TOI","GP_greater_than_0"]

#Drop columns as given on course website, returns new dataset
df_train = df_train.drop(list_of_dropped_vars, axis=1)
df_test = df_test.drop(list_of_dropped_vars, axis=1)

#%% Pre process data

""" x_* input variables for training and testing dataset"""

#Training and test Data
x_train = df_train.drop([class_name], axis=1)
x_test = df_test.drop([class_name], axis=1)

#add boolean terms for catagories
x_train, col_list_train = dummy(x_train)
x_test, col_list_test = dummy(x_test)

#add interaction terms for all i, j: xi*xj
#x_train = interactions(x_train)
#x_test = interactions(x_test)

x_train = standardize(x_train, col_list_train)
x_test = standardize(x_test, col_list_test)

""" t_* target value for training and testing dataset"""
t_train = df_train[class_name]
t_test = df_test[class_name]


#Normalize target variable and create fit to perform inverse if
# training on normalized target
x = t_train.values
x = x.reshape((len(x), 1))
t_train_scaler = StandardScaler()
t_train_scaler = t_train_scaler.fit(x)
temp = t_train_scaler.transform(x)
t_train_norm = pd.Series(temp.reshape(-1), index=t_train.index,
                         name=class_name)

x = t_test.values
x = x.reshape((len(x), 1))
t_test_scaler = StandardScaler()
t_test_scaler = t_test_scaler.fit(x)
temp = t_test_scaler.transform(x)
t_test_norm = pd.Series(temp.reshape(-1), index=t_test.index,
                         name=class_name)
# inverse transform
#inversed = t_train_scaler.inverse_transform(t_train_norm)


#Insert w0 term for weight vector matrix
x_train.insert(0, 'w0', np.ones(len(x_train), dtype=np.int))
x_test.insert(0, 'w0', np.ones(len(x_test), dtype=np.int))
#Keras implements bias, use_bias = True by default


#%% Keras model
model = Sequential()

m = int(len(x_train.columns))

#The first layer has _ neurons and expects _ input variables
model.add(Dense(36, input_dim=m, kernel_initializer='normal', 
                bias_initializer='ones', 
                kernel_regularizer=regularizers.l2(10), activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5)) #For regularization
model.add(Dense(12, kernel_initializer='normal', 
                bias_initializer='ones', 
                kernel_regularizer=regularizers.l2(10), activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error',
              optimizer= Adam(lr=0.01, beta_1=0.9, beta_2=0.999, 
                              epsilon=1e-08, decay=0.0)
              , metrics=[squaredErr])

# Visualize using tensorboard
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#%% Run model
X = x_train.values
#y = t_train_norm.values
y = t_train.values
history = model.fit(X, y, validation_split=0.33, epochs=250, batch_size=32, verbose=0, callbacks=None)
#view tensorboard with tensorboard callbacks=[tensorboard]
#%%
#Metrics
scores = model.evaluate(X, y, batch_size=191)
#Evaluates per batch!!!
print("{}: {:0.2f}".format(model.metrics_names[1]+" train", scores[1]))

X_test = x_test.values
#y_test = t_test_norm.values
y_test = t_test.values
scores = model.evaluate(X_test, y_test, batch_size=len(y_test))
#Evaluates per batch!!!
print("{}: {:0.2f}".format(model.metrics_names[1]+" eval", scores[1]))

""" verbose: 0 for no logging to stdout, 
    1 for progress bar logging, 
    2 for one log line per epoch."""

preds = model.predict(X_test)
#inversed_p = t_test_scaler.inverse_transform(preds)
mse = metrics.mean_squared_error(y_test, preds.reshape(-1))
N = len(y_test)
tf_session = backend.get_session()
sse = squaredErr(t_test.values, preds.reshape(-1)).eval(session=tf_session)
rootmse = rmse(t_test.values, preds.reshape(-1)).eval(session=tf_session)

print("{}: {:0.2f}".format("Mean squared Err"+" pred", mse.eval(session=tf_session)))
print("{}: {:0.2f}".format("Sum Squared Err:"+" pred", sse))
print("{}: {:0.2f}".format("Root mean squared Err: ", rootmse))
#%%
    #start tensorboard in cmd tensorboard --logdir=logs/ 
# plot metrics
plt.plot(history.history['squaredErr'])
plt.plot(history.history['val_squaredErr'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()