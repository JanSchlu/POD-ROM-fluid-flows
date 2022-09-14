#!/usr/bin/env python3

import torch as pt
import sys
from functions import *
from params import model_params, p_steps, SVD_modes, data_save
sys.path.append('/home/jan/POD-ROM-fluid-flows/')

train_loss = pt.zeros(10000)

#######################################################################################
# Initialize model
#######################################################################################

model = FirstNN(**model_params)
print()
print(model)

#######################################################################################
# define and save datasets
#######################################################################################

data = pt.load(f"{data_save}modeCoeffBinary.pt")
data= data[:,:SVD_modes]

train_data, y_train, test_data, y_test = dataManipulator(data, SVD_modes, p_steps, "residual") 

InData = pt.cat((train_data, test_data), 0)
InScaler = MinMaxScaler()
InScaler.fit(InData)
train_data_norm = InScaler.scale(train_data)
test_data_norm = InScaler.scale(test_data)

OutData = pt.cat((y_train, y_test), 0)
OutScaler = MinMaxScaler()
OutScaler.fit(OutData)
y_train_norm = OutScaler.scale(y_train)
y_test_norm = OutScaler.scale(y_test)

MinIn, MaxIn = InScaler.save()
pt.save(MinIn, f"{data_save}MinInR.pt")
pt.save(MaxIn, f"{data_save}MaxInR.pt")
MinOut, MaxOut = OutScaler.save()
pt.save(MinOut, f"{data_save}MinOutR.pt")
pt.save(MaxOut, f"{data_save}MaxOutR.pt")
pt.save(train_data_norm, f"{data_save}train_data_norm_R.pt")
pt.save(y_train_norm, f"{data_save}y_train_norm_R.pt")
pt.save(test_data_norm, f"{data_save}test_data_norm_R.pt")
pt.save(y_test_norm, f"{data_save}y_test_norm_R.pt")


#######################################################################################
# training model
#######################################################################################

learnRate = 0.001
epochs = 3000
data_saveR = "run/data/R/"
train_lossR = optimize_model(model, train_data_norm, y_train_norm, epochs, lr=learnRate, save_best=data_saveR)
pt.save(train_lossR, f"{data_save}train_lossR.pt")
