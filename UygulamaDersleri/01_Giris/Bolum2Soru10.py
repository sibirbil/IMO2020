# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:23:24 2020
@author: utkuk
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

# Part a
boston = load_boston()
print(boston.data.shape)
print(boston.DESCR)

bos = pd.DataFrame(boston.data, columns = boston.feature_names)
print(bos.head())
bos['PRICE'] = boston.target
print(bos.head())

# Part b
pd.plotting.scatter_matrix(bos)
print(bos.describe())

indices = np.array(np.where(np.abs(bos.corr())>0.7))
indices = np.unique(indices[0,(indices[0,:] != indices[1,:])])
pd.plotting.scatter_matrix(bos.iloc[:,indices])

# Part c
corrMatris = np.abs(bos.corr().iloc[:,0])
sort = corrMatris.sort_values()
pd.plotting.scatter_matrix(bos.loc[:,['RAD', 'TAX', 'LSTAT', 'NOX', 'INDUS']])

# Part d
pd.plotting.hist_series(bos.loc[:,'CRIM'], bins = 20)
bos.loc[bos['CRIM']>40,:]

pd.plotting.hist_series(bos.loc[:,'TAX'], bins = 20)
bos.loc[bos['TAX']>700,:]

pd.plotting.hist_series(bos.loc[:,'PTRATIO'], bins = 50)
bos.loc[bos['PTRATIO']>22,:]

# Part e
sum(bos['CHAS']==1)

# Part f
bos['PTRATIO'].median()

# Part g
minPrice = np.where(bos['PRICE'] == bos['PRICE'].min())[0]
bos.iloc[minPrice,:]
print(bos.describe())

# Part h
bos.loc[bos['RM']>7,:].shape
bos.loc[bos['RM']>8,:].shape

bos.loc[bos['RM']>7,:]
print(bos.describe())