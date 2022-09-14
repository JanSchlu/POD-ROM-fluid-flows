#!/usr/bin/env python3

import torch as pt
import sys
from functions import *
from params import data_save, model_params
sys.path.append('/home/jan/POD-ROM-fluid-flows/')
                                             
lenTest = pt.load(f"{data_save}lenTest.pt")
test_data_norm_S = pt.load(f"{data_save}test_data_norm_S.pt")
test_data_norm_R = pt.load(f"{data_save}test_data_norm_R.pt")
test_data_norm_BW = pt.load(f"{data_save}test_data_norm_BW.pt")
MinInS = pt.load(f"{data_save}MinInS.pt")
MinInS = MinInS[:SVD_modes]
MaxInS = pt.load(f"{data_save}MaxInS.pt")
MaxInS = MaxInS[:SVD_modes]
MinOutS = pt.load(f"{data_save}MinOutS.pt")
MinOutS = MinOutS[:SVD_modes]
MaxOutS = pt.load(f"{data_save}MaxOutS.pt")
MaxOutS = MaxOutS[:SVD_modes]
MinInR = pt.load(f"{data_save}MinInR.pt")
MinInR = MinInR[:SVD_modes]
MaxInR = pt.load(f"{data_save}MaxInR.pt")
MaxInR = MaxInR[:SVD_modes]
MinOutR = pt.load(f"{data_save}MinOutR.pt")
MinOutR = MinOutR[:SVD_modes]
MaxOutR = pt.load(f"{data_save}MaxOutR.pt")
MaxOutR = MaxOutR[:SVD_modes]
MinOutBW = pt.load(f"{data_save}MinOutBW.pt")
MinOutBW = MinOutBW[:SVD_modes]
MaxOutBW = pt.load(f"{data_save}MaxOutBW.pt")
MaxOutBW = MaxOutBW[:SVD_modes]
MinInBW = pt.load(f"{data_save}MinInBW.pt")
MinInBW = MinInBW[:SVD_modes]
MaxInBW = pt.load(f"{data_save}MaxInBW.pt")
MaxInBW = MaxInBW[:SVD_modes]
######################################################################################
# load model
#######################################################################################

data_saveS = "/home/jan/POD-ROM-fluid-flows/run/data/S/"
best_modelS = FirstNN(**model_params)
best_modelS.load_state_dict(pt.load(f"{data_saveS}best_model_train.pt"))
data_saveR = "/home/jan/POD-ROM-fluid-flows/run/data/R/"
best_modelR = FirstNN(**model_params)
best_modelR.load_state_dict(pt.load(f"{data_saveR}best_model_train.pt"))
data_saveBW = "/home/jan/POD-ROM-fluid-flows/run/data/BW/"
best_modelBW = FirstNN(**model_params)
best_modelBW.load_state_dict(pt.load(f"{data_saveBW}best_model_train.pt"))

#######################################################################################
# one timestep prediction
#######################################################################################

predS = predictor_sequential(best_modelS, test_data_norm_S)           # alle test Datensätze sind gleich
predR = predictor_residual(best_modelR, test_data_norm_R)
predBW = predictor_backward(best_modelBW, test_data_norm_BW, MinInBW, MaxInBW, MinOutBW, MaxOutBW)

#######################################################################################
# predict from predicted
#######################################################################################


predS_period = predictor_sequential_period(best_modelS, test_data_norm_S, MinInS, MaxInS, MinOutS, MaxOutS)
predR_period = predictor_residual_period(best_modelR, test_data_norm_R, MinInR, MaxInR, MinOutR, MaxOutR)
#predBW_period= predictor_backward_period(best_modelBW, test_data_norm_BW, MinOutBW, MaxOutBW)

pt.save(predS,f"{data_save}predS.pt")
pt.save(predR,f"{data_save}predR.pt")
pt.save(predBW,f"{data_save}predBW.pt")
pt.save(predS_period,f"{data_save}predS_period.pt")
pt.save(predR_period,f"{data_save}predR_period.pt")
#pt.save(predBW_period,f"{data_save}predBW_period.pt")