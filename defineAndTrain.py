#!/usr/bin/env python3

import torch as pt
from functions import *
from params import model_params, p_steps, SVD_modes, data_save

train_loss = pt.zeros(10000)

learnRate = 0.001
epochs = 3000

modeCoeff = pt.load(f"{data_save}modeCoeff.pt")
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
test_data, y_test = rearrange_data(split_data(modeCoeff, lenTest, SVD_modes, lenTrain), p_steps)
pt.save(test_data, f"{data_save}test_data.pt")
pt.save(y_test, f"{data_save}y_test.pt")

#######################################################################################
# training model
#######################################################################################

train_loss = optimize_model(model, train_data, y_train, epochs, lr=learnRate, save_best=data_save)

pt.save(train_loss, f"{data_save}train_loss.pt")