#!/usr/bin/env python3

import torch as pt
import sys
sys.path.append('/home/jan/POD-ROM-fluid-flows/')
from functions import *
from params import model_params, p_steps,data_save

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

#modeCoeff = pt.load(f"{data_save}modeCoeff.pt")
data1 = pt.tensor([[50, 0.2],[50, 0.3],[50, 0.4],[50, 0.5],[50, 0.6],[50, 0.7],[50, 0.8],[50, 0.9],[50, 0.10],[50, 0.11]])
train_data1, y_train1, test_data1, y_test1 = dataManipulator(data1, SVD_modes, p_steps, "sequential")                    #!!!!!! minMax wird nur einmal gespeichert nicht für jede Re
data2 = pt.tensor([[222, 2],[222, 4],[222, 6],[222, 8],[222, 10],[222, 12],[222, 14],[222, 16],[222, 18],[222, 20]])
train_data2, y_train2, test_data2, y_test2 = dataManipulator(data2, SVD_modes, p_steps, "sequential")                    #!!!!!! minMax wird nur einmal gespeichert nicht für jede Re
#data3 = pt.tensor([[333, 4],[333, 8],[333, 12],[333, 16],[333, 20],[333, 24],[333, 28],[333, 32],[333, 36],[333, 40]])
#train_data3, y_train3, test_data3, y_test3 = dataManipulator(data3, SVD_modes, p_steps)                    #!!!!!! minMax wird nur einmal gespeichert nicht für jede Re


train_data = pt.cat((train_data1, train_data2),0)
y_train = pt.cat((y_train1, y_train2), 0)
test_data1 = test_data1[:-1]
test_data2 = test_data2[:-1]
#test_data3 = test_data3[:-1]
test_data = pt.cat((test_data1,test_data2), 0)
y_test= pt.cat((y_test1,y_test2), 0)


minMaxData = pt.cat((train_data,y_train,test_data,y_test),0)

minData = minMaxData.min(dim=0).values
maxData = minMaxData.max(dim=0).values

train_data = (train_data - minData)/(maxData-minData)
y_train = (y_train - minData)/(maxData-minData)
test_data = (test_data - minData)/(maxData-minData)
y_test = (y_test - minData)/(maxData-minData)

pt.save(minData, f"{data_save}minCoeff.pt")
pt.save(maxData, f"{data_save}maxCoeff.pt")

#######################################################################################
# training model
#######################################################################################

learnRate = 0.005
epochs = 3000

train_loss = optimize_model(model, train_data, y_train, epochs, lr=learnRate, save_best=data_save)

pt.save(train_loss, f"{data_save}train_loss.pt")
pt.save(test_data, f"{data_save}test_data.pt")
pt.save(y_train, f"{data_save}y_trainS.pt")
pt.save(y_test, f"{data_save}y_test.pt")