#!/usr/bin/env python3

import torch as pt
import pickle
import sys
sys.path.append('/home/jan/POD-ROM-fluid-flows/')
from functions import *
from params import model_params, data_save, ReInput

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
#data = pt.load(f"{data_save}modeCoeff.pt")
lenDataset = int(len(data)/3)

data1= data#[:lenDataset,:]
data2= data[lenDataset:lenDataset*2,:]
data3= data[lenDataset*2:,:]
train_data1, y_train1, test_data1, y_test1 = dataManipulator(data1, 5e-3, "backward")
train_data2, y_train2, test_data2, y_test2 = dataManipulator(data2, 5e-3, "backward")  
train_data3, y_train3, test_data3, y_test3 = dataManipulator(data3, 5e-3, "backward")  

train_data = train_data1#pt.cat((train_data1, train_data2, train_data3), 0)
test_data = test_data1 #pt.cat((test_data1, test_data2, test_data3), 0)
y_train = y_train1 #pt.cat((y_train1, y_train2, y_train3), 0)
y_test = y_test1 #pt.cat((y_test1, y_test2, y_test2), 0)   

InData = pt.cat((train_data, test_data), 0)
InScaler = MinMaxScaler()
InScaler.fit(InData, ReInput)
train_data_norm = InScaler.scale(train_data)
test_data_norm = InScaler.scale(test_data)

OutData = pt.cat((y_train, y_test), 0)
OutScaler = MinMaxScaler()
OutScaler.fit(OutData, ReInput)
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