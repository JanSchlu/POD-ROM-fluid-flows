#!/usr/bin/env python3

import torch as pt
import pickle
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

train_data, y_train, test_data, y_test = dataManipulator(data, SVD_modes, p_steps, "sequential") 

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


#MinInS = MinInS[:SVD_modes]
#MaxInS = MaxInS[:SVD_modes]
#MinOutS = MinOutS[:SVD_modes]
#MaxOutS = MaxOutS[:SVD_modes]

scalerdict = dataloader(f"{data_save}scalerdict.pkl")
scalerdict["MinInS"], scalerdict["MaxInS"] = InScaler.save()
scalerdict["MinOutS"], scalerdict["MaxOutS"] = OutScaler.save()
with open(f"{data_save}scalerdict.pkl", "wb") as output:
    pickle.dump(scalerdict,output)

pt.save(train_data_norm, f"{data_save}train_data_norm_S.pt")
pt.save(y_train_norm, f"{data_save}y_train_norm_S.pt")
pt.save(test_data_norm, f"{data_save}test_data_norm_S.pt")
pt.save(y_test_norm, f"{data_save}y_test_norm_S.pt")

#######################################################################################
# training model
#######################################################################################
learnRate = 0.001
epochs = 3000

data_saveS = "run/data/S/"
train_loss = optimize_model(model, train_data_norm, y_train_norm, epochs, lr=learnRate, save_best=data_saveS)
pt.save(train_loss, f"{data_save}train_loss.pt")
