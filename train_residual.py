#!/usr/bin/env python3

import torch as pt
import sys
from functions import *
from params import model_params, p_steps, SVD_modes, data_save
sys.path.append('/home/jan/POD-ROM-fluid-flows/')

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
train_data1, y_train1, test_data1, y_test1 = dataManipulator(data1, SVD_modes, p_steps, "residual")                    #!!!!!! minMax wird nur einmal gespeichert nicht für jede Re
data2 = pt.load(f"{data_save}modeCoeff142.pt")
train_data2, y_train2, test_data2, y_test2 = dataManipulator(data2, SVD_modes, p_steps, "residual")                    #!!!!!! minMax wird nur einmal gespeichert nicht für jede Re
data3 = pt.load(f"{data_save}modeCoeff302.pt")
train_data3, y_train3, test_data3, y_test3 = dataManipulator(data3, SVD_modes, p_steps, "residual")  

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

pt.save(minData, f"{data_save}min_y_trainR.pt")
pt.save(maxData, f"{data_save}mAX_y_trainR.pt")


#######################################################################################
# training model
#######################################################################################

learnRate = 0.001
epochs = 3000
data_saveR = "run/data/R/"
train_lossR = optimize_model(model, train_data, y_train, epochs, lr=learnRate, save_best=data_saveR)
pt.save(train_lossR, f"{data_save}train_lossR.pt")
