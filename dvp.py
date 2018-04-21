# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:57:34 2018

@author: Vijay Anand
"""

import numpy as np
import pandas as pd
import os
import time
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras import optimizers
from keras.models import load_model
from keras2pmml import keras2pmml



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
# GRU model
#==============================================================================

def init_gru_model(trainLen,predLen,layers,cells,learningRate=0.00001):
    
        model = Sequential()
        
        model.add(GRU(trainLen, batch_input_shape=(1, 1, trainLen), stateful=True,
                       return_sequences=True ))
        
        for layer in range(layers-2):
            
            model.add(GRU(cells,stateful=True,return_sequences=True))
            
        model.add(GRU(cells,stateful=True,return_sequences=True))
        model.add(GRU(predLen,stateful=True,return_sequences=False))
            
#        model.add(Dense(predLen))
        
        o=optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999)
        model.compile(loss='mean_squared_error', optimizer=o)
        
        model.summary()
        
        
        return model
        
        
#==============================================================================
# Train model
#==============================================================================

    
def train_model(model,train,val,numEpochs,trainLen,predLen):
    
    trainData,trainLabel=train[:,:trainLen],train[:,trainLen:]
    trainData=trainData.reshape(trainData.shape[0],1,trainData.shape[1])
    
    for epoch in range(numEpochs):
        
        history=model.fit(trainData, trainLabel, epochs=1, batch_size=1, verbose=1,
                  shuffle=False)
                  
            
#        trainingLoss=history['loss']
#        trainingAccuracy=history['acc']
#        valLoss=history['val_loss']
#        valAccuracy=history['val_acc']
#        
        model.reset_states()
        
        validate(model,val,trainLen)
        
        
    return model
    
#==============================================================================
# Val
#==============================================================================

def validate(model,val,predLen):

    valData,valLabel=val[:,:predLen],val[:,predLen:]
    valData=valData.reshape(valData.shape[0],1,valData.shape[1])
    
    results=model.predict(valData,batch_size=1)
    
    actual=valLabel

    error=results-actual
    
    error=np.square(error)
    error=np.sqrt(error)
    rmse=np.mean(error,axis=0)
    
    print ('==============Results==============')
    print ('RMSE: ',rmse*100)
    print ('Average MSE:' ,np.mean(rmse)*100)

#==============================================================================
# def main
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
    
    model=init_gru_model(trainLen,predLen,layers=3,cells=64,learningRate=0.000001)
#    trainedModel=train_model(model,train,val,5,trainLen,predLen)
#    
#    trainedModel.save('stocks.h5')
#    
#    ####
#    df1=pd.DataFrame(val[:,:trainLen],columns=['Day -'+str(x) for x in range(9,-1,-1)])
#    df2=pd.DataFrame(val[:,trainLen:],columns=['Day +'+str(x) for x in range(1,6)])
#    
#    df1.to_csv('input.csv',index=None)
#    df2.to_csv('actuals.csv',index=None)
    
#    keras2pmml(estimator=trainedModel,file='keras_iris.pmml')
    
    output = open('myfile.pkl', 'wb')
    pickle.dump(model, output)
    output.close()

if __name__=='__main__':

    main()