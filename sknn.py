# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:17:10 2018

@author: Vijay Anand
"""

from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import os
import time
import pickle

#==============================================================================
# Generate data for model
#==============================================================================

def gen_data(lst,size):

    newList=[]

    for indx in range(len(lst)):

        if len(lst)-indx>size:

            newList.append(np.array(lst[indx:indx+size]))


    newFeats=np.vstack(newList)

    return newFeats
#==============================================================================
# main
#==============================================================================
def main():
    
    trainLen=10
    predLen=5

    df=pd.read_csv('daily-foreign-exchange-rates-31-.csv')

    df=df.sort_values(by=['Date'])

    feats=df['Exchange rate'].tolist()

    train=feats[:int(len(feats)*0.8)]
    val=feats[int(len(feats)*0.8):]

    train=gen_data(train,15)
    val=gen_data(val,15)
    
    ####
    
    print (train.shape,val.shape)
    
    reg=MLPRegressor(solver='adam', alpha=1e-5,hidden_layer_sizes=(trainLen,128,128,128,predLen),
                     learning_rate_init=0.001,verbose=True,random_state=1)
                     
    reg.fit(train[:,:trainLen],train[:,trainLen:])
    
    preds=reg.predict(val[:,:trainLen])
    actuals=val[:,trainLen:]
    
    print (preds.shape)
    
    error=preds-actuals
        
    error=np.square(error)
    error=np.sqrt(error)
    rmse=np.mean(error,axis=0)
    
    print ('==============Results==============')
    print ('RMSE: ',rmse*100)
    print ('Average MSE:' ,np.mean(rmse)*100)

#    
#    ####
#    df1=pd.DataFrame(val[:,:trainLen],columns=['Day -'+str(x) for x in range(9,-1,-1)])
#    df2=pd.DataFrame(val[:,trainLen:],columns=['Day +'+str(x) for x in range(1,6)])
#    
#    df1.to_csv('input.csv',index=None)
#    df2.to_csv('actuals.csv',index=None)
    
#    keras2pmml(estimator=trainedModel,file='keras_iris.pmml')
    
    output = open('myfile.pkl', 'wb')
    pickle.dump(reg, output)
    output.close()

if __name__=='__main__':

    main()