#!/usr/bin/env python3

import torch as pt
import sys
sys.path.append('/home/jan/POD-ROM-fluid-flows/')
from functions import *
from params import model_params, p_steps, SVD_modes, data_save

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

train_data, y_train, test_data, y_test = dataManipulator(data, SVD_modes, p_steps, "backward")

InData = pt.cat((train_data, test_data), 0)
InScaler = MinMaxScaler()
InScaler.fit(InData)
train_data_norm = InScaler.scale(train_data)
test_data_norm = InScaler.scale(test_data)
InScaler.fit(data)

OutData = pt.cat((y_train, y_test), 0)
OutScaler = MinMaxScaler()
OutScaler.fit(OutData)
y_train_norm = OutScaler.scale(y_train)
y_test_norm = OutScaler.scale(y_test)

scalerdict = dataloader(f"{data_save}scalerdict.pkl")
scalerdict["MinInBW"], scalerdict["MaxInBW"] = InScaler.save()
scalerdict["MinOutBW"], scalerdict["MaxOutBW"] = OutScaler.save()
with open(f"{data_save}scalerdict.pkl", "wb") as output:
    pickle.dump(scalerdict,output)

pt.save(test_data_norm, f"{data_save}test_data_norm.pt")
pt.save(y_test_norm, f"{data_save}y_test_norm_BW.pt")

#######################################################################################
# training model
#######################################################################################

learnRate = 0.001
epochs = 3000
data_saveBW = "run/data/BW/"

train_lossBW = optimize_model(model, train_data_norm, y_train_norm, epochs, lr=learnRate, save_best=data_saveBW)
pt.save(train_lossBW, f"{data_save}train_lossBW.pt")