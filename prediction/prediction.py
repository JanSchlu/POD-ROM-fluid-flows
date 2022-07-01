#!/usr/bin/env python3

import torch as pt
import sys
sys.path.append('/home/jan/POD-ROM-fluid-flows/')
from functions import *
from params import data_save, SVD_modes, p_steps, model_params
                                             
lenTest = pt.load(f"{data_save}lenTest.pt")
test_data = pt.load(f"{data_save}test_data.pt")
#y_test = pt.load(f"{data_save}y_test.pt")
#y_testR = pt.load(f"{data_save}y_testBW.pt")
#y_testBW = pt.load(f"{data_save}y_testBW.pt")

######################################################################################
# load model
#######################################################################################

best_model = FirstNN(**model_params)
best_model.load_state_dict(pt.load(f"{data_save}best_model_train.pt"))
data_saveR = "/home/jan/POD-ROM-fluid-flows/run/data/R/"
best_modelR = FirstNN(**model_params)
best_modelR.load_state_dict(pt.load(f"{data_saveR}best_model_train.pt"))
data_saveBW = "/home/jan/POD-ROM-fluid-flows/run/data/BW/"
best_modelBW = FirstNN(**model_params)
best_modelBW.load_state_dict(pt.load(f"{data_saveBW}best_model_train.pt"))

#######################################################################################
# one timestep prediction
#######################################################################################
def predictor(data, prediction_update):
    predict = pt.ones([len(data)-p_steps,SVD_modes])                                           # pred len(test_data)-1-p_steps
    for i in range (0, len(predict)):
        if prediction_update == "sequential":
            predict[i] = best_model(data[i]).squeeze()
        if prediction_update == "residual":
            if i>1:
                predict[i] = data[i-1,p_steps * 20:] + best_modelR(data[i]).squeeze()
        if prediction_update == "backward":
            if i>2:
                predict[i] = 4/3*predict[i-1] - 1/3*predict[i-2] + 2/3*best_modelBW(data[i]).squeeze()*5e-3
    predict = predict.detach().numpy()
    return predict
predS = predictor(test_data, "sequential")    
predR = predictor(test_data, "residual")
predBW = predictor(test_data, "backward")

#######################################################################################
# predict from predicted
#######################################################################################
def preditorOfPredicted(data, scheme):
    predStore = pt.ones([len(data)-p_steps,SVD_modes+SVD_modes*p_steps])                 
    predicted = pt.ones([len(data)-p_steps,SVD_modes])
    predStore[0] = data[0]                                                               #start is last timestep of trainData
    predicted[0] = data[0, SVD_modes*p_steps:]
    for i in range (1, len(predStore)):
        if scheme == "sequential":
            prediction = best_model(predStore[i-1]).squeeze()
        if scheme == "residual":
            prediction = predStore[i-1,p_steps*20:]+best_modelR(predStore[i-1]).squeeze()
        if scheme == "backward":
            prediction = 4/3*predStore[i-1,p_steps*20:] - 1/3*predStore[i-1,p_steps*10:p_steps*10+10] + 2/3*best_modelBW(predStore[i-1]).squeeze()*5e-3
        predStore[i] = pt.cat((predStore[i-1,SVD_modes:], prediction))
        predicted[i-1] = prediction 
    predicted = predicted.detach().numpy()
    return predicted

predS_period = preditorOfPredicted(test_data, "sequential")
print(predS_period)
predR_period = preditorOfPredicted(test_data, "residual")
print(predR_period)
predBW_period= preditorOfPredicted(test_data, "backward")
print(predBW_period)

pt.save(predS,f"{data_save}predS.pt")
pt.save(predR,f"{data_save}predR.pt")
pt.save(predS_period,f"{data_save}predS_period.pt")
pt.save(predR_period,f"{data_save}predR_period.pt")
pt.save(predBW,f"{data_save}predBW.pt")
pt.save(predBW_period,f"{data_save}predBW_period.pt")