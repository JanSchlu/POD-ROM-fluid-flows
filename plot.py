#!/usr/bin/env python3
'''
plot

'''
from pickle import TRUE
from re import T
from matplotlib import pyplot as plt
import torch as pt
from functions import *
from params import data_save, SVD_modes, ReInput

# increase plot resolution
plt.rcParams["figure.dpi"] = 500

plt_save = "/home/jan/POD-ROM-fluid-flows/run/plot/"                   

train_loss = pt.load(f"{data_save}train_loss.pt")
train_lossR = pt.load(f"{data_save}train_lossR.pt")
train_lossBW = pt.load(f"{data_save}train_lossBW.pt")

test_data_norm_S = pt.load(f"{data_save}test_data_norm.pt")
y_test_norm_S = pt.load(f"{data_save}y_test_norm_S.pt")
y_test_norm_R = pt.load(f"{data_save}y_test_norm_R.pt")
y_test_norm_BW = pt.load(f"{data_save}y_test_norm_BW.pt") 

predS_period = pt.load(f"{data_save}predS_period.pt")
predR_period = pt.load(f"{data_save}predR_period.pt")
predBW_period = pt.load(f"{data_save}predBW_period.pt")

scalerdict = dataloader(f"{data_save}scalerdict.pkl")
InScalerS = MinMaxScaler()
InScalerS.refit(scalerdict["MinInS"], scalerdict["MaxInS"], ReInput)
OutScalerS = MinMaxScaler()
OutScalerS.refit(scalerdict["MinOutS"], scalerdict["MaxOutS"], ReInput)  
InScalerR = MinMaxScaler()
InScalerR.refit(scalerdict["MinInR"], scalerdict["MaxInR"], ReInput)
OutScalerR = MinMaxScaler()
OutScalerR.refit(scalerdict["MinOutR"], scalerdict["MaxOutR"], ReInput)  
InScalerBW = MinMaxScaler()
InScalerBW.refit(scalerdict["MinInBW"], scalerdict["MaxInBW"], ReInput)
OutScalerBW = MinMaxScaler()
OutScalerBW.refit(scalerdict["MinOutBW"], scalerdict["MaxOutBW"], ReInput)  

predS_period = pt.tensor(predS_period)
predR_period = pt.tensor(predR_period)
predBW_period = pt.tensor(predBW_period)

#######################################################################################
# plot training/validation loss
#######################################################################################

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(1, epochs+1), train_loss,  'g', lw=1.0, label="training loss S")
plt.plot(range(1, epochs+1), train_lossR, 'r', lw=1.0, label="training loss R")
plt.plot(range(1, epochs+1), train_lossBW,'b', lw=1.0, label="training loss BW")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.xlim(1, epochs+1)
plt.yscale("log")
plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}loss.png")

#######################################################################################
# plot prediction
#######################################################################################

test_dataS = InScalerS.rescale(test_data_norm_S)
predS_period = InScalerS.rescale(predS_period)
predR_period = InScalerR.rescale(predR_period)
predBW_period = InScalerS.rescale(predBW_period)
y_testS = OutScalerS.rescale(y_test_norm_S)

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
if ReInput == 1:
    count = 1
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(y_testS)), y_testS[:,count],           'b',    lw=1, label=f"S test y")
        axarr[row, col].plot(range(0 , len(test_dataS)), test_dataS[:,count],           'k',    lw=1, label=f"S test x")        
        axarr[row, col].plot(range(0 , len(predS_period)), predS_period[:,count],  'g', ls=':', lw=2, label=f"Vorhersage S")           
        axarr[row, col].grid()
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, len(test_dataS))
plt.ylim(-0.5,0.5)
plt.legend()
plt.savefig(f"{plt_save}PredictionS.png")

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
if ReInput == 1:
    count = 1
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(test_dataS)), test_dataS[:,count],           'k',    lw=1, label=f"R test x")
        axarr[row, col].plot(range(0 , len(predR_period)), predR_period[:,count],  'r', ls=':', lw=2, label=f"Vorhersage R")           
        axarr[row, col].grid()
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, len(test_dataS))
plt.ylim(-0.5,0.5)
plt.legend()
plt.savefig(f"{plt_save}PredictionR.png")

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
if ReInput == 1:
    count = 1
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(test_dataS)), test_dataS[:,count],           'k',    lw=1, label=f"BW test x")
        axarr[row, col].plot(range(0 , len(predBW_period)), predBW_period[:,count],  'b', ls=':', lw=2, label=f"Vorhersage BW")           
        axarr[row, col].grid()
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, len(test_dataS))
plt.ylim(-0.5,0.5)
plt.legend()
plt.savefig(f"{plt_save}PredictionBW.png")

#######################################################################################
# error plot
#######################################################################################

if ReInput == True:
    test_dataS = test_dataS[:,1:]
    predS_period = predS_period[:,1:]
    predR_period = predR_period[:,1:]
    predBW_period = predBW_period[:,1:]

errSeq = (test_dataS-predS_period)**2
meanErrSeq = pt.sum(errSeq,1)/SVD_modes
errSeq = errSeq.detach().numpy()
meanErrSeq = meanErrSeq.detach().numpy()

errRSeq = (test_dataS-predR_period)**2
meanErrRSeq = pt.sum(errRSeq,1)/SVD_modes
errRSeq = errRSeq.detach().numpy()
meanErrRSeq = meanErrRSeq.detach().numpy()

errBWSeq = (test_dataS-predBW_period)**2
meanErrbwSeq = pt.sum(errBWSeq,1)/SVD_modes
errBWSeq = errBWSeq.detach().numpy()
meanErrbwSeq = meanErrbwSeq.detach().numpy()

#######################################################################################
# plot mean error plot
#######################################################################################

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(0, len(meanErrbwSeq)), meanErrSeq, 'g', lw=1.0, label= "seq. prediction error S")
plt.plot(range(0, len(meanErrbwSeq)), meanErrRSeq, 'r', lw=1.0, label= "seq. prediction error R")
plt.plot(range(0, len(meanErrbwSeq)), meanErrbwSeq, 'b', lw=1.0, label= "seq. prediction error BW")
plt.xlim(0, len(meanErrSeq))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError.png")