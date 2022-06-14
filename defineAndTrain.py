#!/usr/bin/env python3
"""
code for initialisation of the neural network and training it
"""
from optparse import Values
import torch as pt
from functions import *
from params import model_params, p_steps, SVD_modes, data_save

train_loss = pt.zeros(10000)

learnRate = 0.005
epochs = 3000

modeCoeff = pt.load(f"{data_save}modeCoeff.pt")
#modeCoeff = subtract_data(modeCoeff)
minCoeff = modeCoeff.min(dim=0).values
maxCoeff = modeCoeff.max(dim=0).values
modeCoeff = (modeCoeff - minCoeff)/(maxCoeff-minCoeff)
pt.save(minCoeff, f"{data_save}minCoeff.pt")
pt.save(maxCoeff, f"{data_save}maxCoeff.pt")
window_times = pt.load(f"{data_save}window_times.pt")

#######################################################################################
# Initialize model
#######################################################################################

model = FirstNN(**model_params)
print()
print(model)

#######################################################################################
# define and save datasets
#######################################################################################

maxLen = (len(modeCoeff))
lenTrain = int(maxLen * 2 / 3)    # training data sind die ersten 2/3 Zeitschritte mit SVD_modes Koeffizienten [batch,Anzahl Coeff]
lenTest = maxLen - lenTrain
pt.save(lenTrain, f"{data_save}lenTrain.pt")
pt.save(lenTest, f"{data_save}lenTest.pt")
print(maxLen, lenTrain, lenTest)
train_data, y_train ,y_trainR, y_trainBW = rearrange_data(split_data(modeCoeff, lenTrain, SVD_modes,0), p_steps)
test_data, y_test, y_testR, y_testBW = rearrange_data(split_data(modeCoeff, lenTest, SVD_modes, lenTrain), p_steps)

print("y_train",y_train.max(dim=0).values - y_train.min(dim=0).values)
print("y_trainR", y_trainR.max(dim=0).values - y_trainR.min(dim=0).values)
print("y_trainBW", y_trainBW.max(dim=0).values - y_trainBW.min(dim=0).values)
pt.save(test_data, f"{data_save}test_data.pt")
pt.save(y_test, f"{data_save}y_test.pt")
print("y_test", y_test.max(dim=0).values - y_test.min(dim=0).values)
pt.save(y_testR, f"{data_save}y_testR.pt")
print("y_testR", y_testR.max(dim=0).values - y_testR.min(dim=0).values)
pt.save(y_testBW, f"{data_save}y_testBW.pt")
print("y_testBW", y_testBW.max(dim=0).values - y_testBW.min(dim=0).values)
#######################################################################################
# training model
#######################################################################################


train_loss = optimize_model(model, train_data, y_train, epochs, lr=learnRate, save_best=data_save)
pt.save(train_loss, f"{data_save}train_loss.pt")
data_saveR = "data/R/"
train_lossR = optimize_model(model, train_data, y_trainR, epochs, lr=learnRate, save_best=data_saveR)
pt.save(train_lossR, f"{data_save}train_lossR.pt")
data_saveBW = "data/BW/"
train_lossBW = optimize_model(model, train_data, y_trainBW, epochs, lr=learnRate, save_best=data_saveBW)
pt.save(train_lossBW, f"{data_save}train_lossBW.pt")