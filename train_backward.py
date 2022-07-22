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
y_trainBW = recalculate_output(train_data,y_train,"backward",5e-3)
train_data=train_data[:-1]


#######################################################################################
# training model
#######################################################################################


learnRate = 0.01
epochs = 30000
print(train_data[0],train_data[1],train_data[2],train_data[3],train_data[4],train_data[5],train_data[6])
print(y_trainBW)
data_saveBW = "run/data/BW/"
train_lossBW = optimize_model(model, train_data, y_trainBW, epochs, lr=learnRate, save_best=data_saveBW)
pt.save(train_lossBW, f"{data_save}train_lossBW.pt")

test_data, y_test = rearrange_data(split_data(modeCoeff, lenTest, SVD_modes, lenTrain), p_steps)

y_testBW , outforprintBW = recalculate_output(test_data,y_test,"backward",5e-3)
test_data = test_data[:-1]


pt.save(outforprintBW, f"{data_save}outforprintBW.pt")
pt.save(y_testBW, f"{data_save}y_testBW.pt")
pt.save(y_trainBW, f"{data_save}y_trainBW.pt")
pt.save(test_data, f"{data_save}test_data.pt")

## problem: y_trainBW schwankt zwischen -9 und 9 (residual schwankt zwischen -0,4 und 0,4, sequential zwischen 0 und 1)