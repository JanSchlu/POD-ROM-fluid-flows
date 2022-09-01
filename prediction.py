#!/usr/bin/env python3

import torch as pt
import sys
from functions import *
from params import data_save, model_params
sys.path.append('/home/jan/POD-ROM-fluid-flows/')
                                             
lenTest = pt.load(f"{data_save}lenTest.pt")
test_data = pt.load(f"{data_save}test_data.pt")
print(test_data.shape)
min_y_train = pt.load(f"{data_save}min_y_trainS.pt")
max_y_train = pt.load(f"{data_save}max_y_trainS.pt")
min_y_trainR = pt.load(f"{data_save}min_y_trainR.pt")
max_y_trainR = pt.load(f"{data_save}max_y_trainR.pt")
min_y_trainBW = pt.load(f"{data_save}min_y_trainBW.pt")
max_y_trainBW = pt.load(f"{data_save}max_y_trainBW.pt")

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

predS = predictor_sequential(best_model, test_data, min_y_train, max_y_train)
predR = predictor_residual(best_modelR, test_data, min_y_trainR, max_y_trainR)
predBW = predictor_backward(best_modelBW, test_data, min_y_trainBW, max_y_trainBW)

print(predS.shape)
print(predR.shape)
print(predBW.shape)
#######################################################################################
# predict from predicted
#######################################################################################


#predS_period = predictor_sequential_period(best_model, test_data, min_y_train, max_y_train)
#predR_period = predictor_residual_period(best_modelR, test_data, min_y_trainR, max_y_trainR)
#predBW_period= predictor_backward_period(best_modelBW, test_data, min_y_trainBW, max_y_trainBW)

pt.save(predS,f"{data_save}predS.pt")
pt.save(predR,f"{data_save}predR.pt")
pt.save(predBW,f"{data_save}predBW.pt")
#pt.save(predS_period,f"{data_save}predS_period.pt")
#pt.save(predR_period,f"{data_save}predR_period.pt")
#pt.save(predBW_period,f"{data_save}predBW_period.pt")