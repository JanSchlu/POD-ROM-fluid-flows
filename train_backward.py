#!/usr/bin/env python3

import torch as pt
import sys
sys.path.append('/home/jan/POD-ROM-fluid-flows/')
from functions import *
from params import model_params, p_steps, SVD_modes, data_save

train_loss = pt.zeros(10000)

modeCoeff = pt.load(f"{data_save}modeCoeff.pt")
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

data1 = pt.load(f"{data_save}modeCoeff56.pt")
train_data1, y_train1, test_data1, y_test1 = dataManipulator(data1, SVD_modes, p_steps, "backward")                    #!!!!!! minMax wird nur einmal gespeichert nicht für jede Re
data2 = pt.load(f"{data_save}modeCoeff142.pt")
train_data2, y_train2, test_data2, y_test2 = dataManipulator(data2, SVD_modes, p_steps, "backward")                    #!!!!!! minMax wird nur einmal gespeichert nicht für jede Re
data3 = pt.load(f"{data_save}modeCoeff302.pt")
train_data3, y_train3, test_data3, y_test3 = dataManipulator(data3, SVD_modes, p_steps, "backward")  

train_data = pt.cat((train_data1, train_data2, train_data3),0)
y_train = pt.cat((y_train1, y_train2,y_train3), 0)
test_data1 = test_data1[:-1]
test_data2 = test_data2[:-1]
test_data3 = test_data3[:-1]
test_data = pt.cat((test_data1,test_data2, test_data3), 0)
y_test= pt.cat((y_test1,y_test2, y_test3), 0)

minMaxData = pt.cat((train_data,y_train,test_data,y_test),0)

minData = minMaxData.min(dim=0).values
maxData = minMaxData.max(dim=0).values

train_data = (train_data - minData)/(maxData-minData)
y_train = (y_train - minData)/(maxData-minData)
test_data = (test_data - minData)/(maxData-minData)
y_test = (y_test - minData)/(maxData-minData)

pt.save(minData, f"{data_save}min_y_trainBW.pt")
pt.save(maxData, f"{data_save}mAX_y_trainBW.pt")

#######################################################################################
# training model
#######################################################################################

learnRate = 0.001
epochs = 3000
data_saveBW = "run/data/BW/"
train_lossBW = optimize_model(model, train_data, y_train, epochs, lr=learnRate, save_best=data_saveBW)
pt.save(train_lossBW, f"{data_save}train_lossBW.pt")

## problem: y_trainBW schwankt zwischen -9 und 9 (residual schwankt zwischen -0,4 und 0,4, sequential zwischen 0 und 1) -> für die erste Mode
## in anderen MOden durchaus -42 +42
## -> Lösung min/max Normierung der Koeffizeienten
