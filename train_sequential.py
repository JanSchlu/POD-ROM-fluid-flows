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
y_train = recalculate_output(train_data,y_train,"sequential",5e-3)
train_data=train_data[:-1]


#######################################################################################
# training model
#######################################################################################
learnRate = 0.005
epochs = 3000

train_loss = optimize_model(model, train_data, y_train, epochs, lr=learnRate, save_best=data_save)
pt.save(train_loss, f"{data_save}train_loss.pt")


test_data, y_test = rearrange_data(split_data(modeCoeff, lenTest, SVD_modes, lenTrain), p_steps)
y_test = recalculate_output(test_data,y_test,"sequential",5e-3)
test_data = test_data[:-1]

pt.save(test_data, f"{data_save}test_data.pt")
pt.save(y_train, f"{data_save}y_trainS.pt")
pt.save(y_test, f"{data_save}y_test.pt")

