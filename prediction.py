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
scalerdict = dataloader(f"{data_save}scalerdict.pkl")

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
predBW = predictor_backward(best_modelBW, test_data_norm_BW, scalerdict["MinInBW"], scalerdict["MaxInBW"], scalerdict["MinOutBW"], scalerdict["MaxOutBW"])

#######################################################################################
# predict from predicted
#######################################################################################


predS_period = predictor_sequential_period(best_modelS, test_data_norm_S, scalerdict["MinInS"], scalerdict["MaxInS"], scalerdict["MinOutS"], scalerdict["MaxOutS"])
predR_period = predictor_residual_period(best_modelR, test_data_norm_R, scalerdict["MinInR"], scalerdict["MaxInR"], scalerdict["MinOutR"], scalerdict["MaxOutR"])
#predBW_period= predictor_backward_period(best_modelBW, test_data_norm_BW, MinOutBW, MaxOutBW)

pt.save(predS,f"{data_save}predS.pt")
pt.save(predR,f"{data_save}predR.pt")
pt.save(predBW,f"{data_save}predBW.pt")
pt.save(predS_period,f"{data_save}predS_period.pt")
pt.save(predR_period,f"{data_save}predR_period.pt")
#pt.save(predBW_period,f"{data_save}predBW_period.pt")