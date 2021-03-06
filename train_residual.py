#!/usr/bin/env python3
import torch as pt
import sys
sys.path.append('/home/jan/POD-ROM-fluid-flows/')
from functions import *
from params import model_params, p_steps, SVD_modes, data_save

train_loss = pt.zeros(10000)


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


train_data, y_train = rearrange_data(split_data(modeCoeff, lenTrain, SVD_modes,0), p_steps)
y_trainR = recalculate_output(train_data,y_train,"residual",5e-3)
train_data=train_data[:-1]


#######################################################################################
# training model
#######################################################################################

learnRate = 0.001
epochs = 3000

data_saveR = "run/data/R/"
train_lossR = optimize_model(model, train_data, y_trainR, epochs, lr=learnRate, save_best=data_saveR)
pt.save(train_lossR, f"{data_save}train_lossR.pt")


test_data, y_test = rearrange_data(split_data(modeCoeff, lenTest, SVD_modes, lenTrain), p_steps)
y_testR = recalculate_output(test_data,y_test,"residual",5e-3)
test_data = test_data[:-1]

pt.save(y_testR, f"{data_save}y_testR.pt")
pt.save(y_trainR, f"{data_save}y_trainR.pt")
pt.save(test_data, f"{data_save}test_data.pt")
