#!/usr/bin/env python3
'''
plot

'''
import torch as pt
from functions import *
from params import data_save

plt_save = "/home/jan/POD-ROM-fluid-flows/run/plot/"    

import torch as pt
import sys
from functions import *
from params import data_save, model_params
sys.path.append('/home/jan/POD-ROM-fluid-flows/')

test_data_norm = pt.load(f"{data_save}test_data_norm.pt")

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
# prediction
#######################################################################################

predS_period = predictor_sequential(best_modelS, test_data_norm, scalerdict)
predR_period = predictor_residual(best_modelR, test_data_norm, scalerdict)
predBW_period= predictor_backward(best_modelBW, test_data_norm, scalerdict)

pt.save(predS_period,f"{data_save}predS_period.pt")
pt.save(predR_period,f"{data_save}predR_period.pt")
pt.save(predBW_period,f"{data_save}predBW_period.pt")