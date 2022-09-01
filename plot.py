#!/usr/bin/env python3
'''
plot

'''
from matplotlib import pyplot as plt
import torch as pt
from functions import *
from params import data_save, SVD_modes


# increase plot resolution
plt.rcParams["figure.dpi"] = 500

plt_save = "/home/jan/POD-ROM-fluid-flows/run/plot/"
lenTrain = pt.load(f"{data_save}lenTrain.pt")                                                       
lenTest = pt.load(f"{data_save}lenTest.pt")
test_data = pt.load(f"{data_save}test_data.pt")
y_test = pt.load(f"{data_save}y_test.pt")
y_test = y_test[:-1]
y_trainS = pt.load(f"{data_save}y_trainS.pt")
y_trainR = pt.load(f"{data_save}y_trainR.pt")
y_trainBW = pt.load(f"{data_save}y_trainBW.pt")
window_times = pt.load(f"{data_save}window_times.pt")
train_loss = pt.load(f"{data_save}train_loss.pt")
train_lossR = pt.load(f"{data_save}train_lossR.pt")
train_lossBW = pt.load(f"{data_save}train_lossBW.pt")
predS = pt.load(f"{data_save}predS.pt")
predR = pt.load(f"{data_save}predR.pt")
predS_period = pt.load(f"{data_save}predS_period.pt")
predR_period = pt.load(f"{data_save}predR_period.pt")
predBW = pt.load(f"{data_save}predBW.pt")
predBW_period = pt.load(f"{data_save}predBW_period.pt")
#######################################################################################
# plot training/validation loss
#######################################################################################

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(1, epochs+1), train_loss, lw=1.0, label="training loss")
plt.plot(range(1, epochs+1), train_lossR, lw=1.0, label="training lossR")
plt.plot(range(1, epochs+1), train_lossBW, lw=1.0, label="training lossBW")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.xlim(1, epochs+1)
plt.yscale("log")
plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}loss.png")

#######################################################################################
# plot prediciton mode
#######################################################################################
print(y_test.shape)
print(predS.shape)
print(predR.shape)
print(predBW.shape)
fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(y_test)), y_test[:,count], 'b', lw=0.5, label=f"orig")
        axarr[row, col].plot(range(0 , len(y_test)), predS[:,count], 'r:', lw=0.5, label=f"S")
        axarr[row, col].plot(range(0 , len(y_test)), predR[:,count], 'g', lw=0.5, label=f"R")
        axarr[row, col].plot(range(0 , len(y_test)), predBW[:,count], lw=0.5, label=f"BW")
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, len(y_test))
#plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}prediction.png")

#######################################################################################
# plot prediciton mode
#######################################################################################

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(y_test)), y_test[:,count], 'b', lw=0.5, label=f"orig")
        axarr[row, col].plot(range(0 , len(y_test)), predS_period[:,count], 'r', lw=0.5, label=f"S_seq")
        axarr[row, col].plot(range(0 , len(y_test)), predR_period[:,count], 'g', lw=0.5, label=f"R_seq")
        axarr[row, col].plot(range(0 , len(y_test)), predBW_period[:,count], lw=0.5, label=f"BW_seq")
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, len(y_test))
#plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}prediction_seq.png")

#######################################################################################
# error plot
#######################################################################################

err = pt.zeros(len(test_data))
errSeq = int
fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0

err = (y_test-predS)**2
errSeq = (y_test-predS_period)**2
meanErr = pt.sum(err,1)/SVD_modes
meanErrSeq = pt.sum(errSeq,1)/SVD_modes
err = err.detach().numpy()
errSeq = errSeq.detach().numpy()
meanErr = meanErr.detach().numpy()
meanErrSeq = meanErrSeq.detach().numpy()

errR = (y_test-predR)**2
errRSeq = (y_test-predR_period)**2
meanErrR = pt.sum(errR,1)/SVD_modes
meanErrRSeq = pt.sum(errRSeq,1)/SVD_modes
errR = errR.detach().numpy()
errRSeq = errRSeq.detach().numpy()
meanErrR = meanErrR.detach().numpy()
meanErrRSeq = meanErrRSeq.detach().numpy()

errBW = (y_test-predBW)**2
errBWSeq = (y_test-predBW_period)**2
meanErrbw = pt.sum(errBW,1)/SVD_modes
meanErrbwSeq = pt.sum(errBWSeq,1)/SVD_modes
errBW = errBW.detach().numpy()
errBWSeq = errBWSeq.detach().numpy()
meanErrbw = meanErrbw.detach().numpy()
meanErrbwSeq = meanErrbwSeq.detach().numpy()

for row in range(2):
    for col in range(2):
        #axarr[row, col].plot(times_num, modeCoeff[:,count], lw=1, label=f"coeff. mode {i+1}")
        axarr[row, col].plot(range(0 , len(y_test)), err[:,count], 'g', lw=1, label=f"coeff. mode {count+1}")
        axarr[row, col].plot(range(0 , len(y_test)), errSeq[:,count], 'r', lw=1, label=f"coeff. mode {count+1}")

        axarr[row, col].plot(range(0 , len(y_test)), errR[:,count], 'g:', lw=1, label=f"coeff. mode {count+1}")
        axarr[row, col].plot(range(0 , len(y_test)), errRSeq[:,count], 'r:', lw=1, label=f"coeff. mode {count+1}")

        axarr[row, col].plot(range(0 , len(y_test)), errBW[:,count], 'g--', lw=1, label=f"coeff. mode {count+1}")
        axarr[row, col].plot(range(0 , len(y_test)), errBWSeq[:,count], 'r--', lw=1, label=f"coeff. mode {count+1}")
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1} error")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.yscale("log")
plt.ylim(bottom=10e-10)
plt.xlim(0, len(y_test))
plt.savefig(f"{plt_save}error.png")

#######################################################################################
# plot mean error plot
#######################################################################################

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(0, len(y_test)), meanErr, 'g', lw=1.0, label="prediction error S")
plt.plot(range(0, len(y_test)), meanErrSeq, 'r', lw=1.0, label= "seq. prediction error S")
plt.xlim(0, len(y_test))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError_S.png")

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(0, len(y_test)), meanErrR, 'g', lw=1.0, label="prediction error R")
plt.plot(range(0, len(y_test)), meanErrRSeq, 'r', lw=1.0, label= "seq. prediction error R")
plt.xlim(0, len(y_test))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError_res.png")

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(0, len(y_test)), meanErrbw, 'g', lw=1.0, label="prediction error BW")
plt.plot(range(0, len(y_test)), meanErrbwSeq, 'r', lw=1.0, label= "seq. prediction error BW")
plt.xlim(0, len(y_test))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError_bw.png")

#######################################################################################
# plot y_train
#######################################################################################
"""
fig, ax = plt.subplots()                                      
plt.plot(range(0, len(y_trainR)), y_trainR[:,0], 'g', lw=1.0, label="R")
plt.plot(range(0, len(y_trainS)), y_trainS[:,0], 'b', lw=1.0, label="S")
plt.plot(range(0, len(y_trainBW)), y_trainBW[:,0], 'r', lw=1.0, label="BW")
plt.xlim(0, len(y_trainR))
plt.xlabel("steps")
plt.ylabel("")
plt.legend()
plt.savefig(f"{plt_save}y_train.png")


for row in range(2):
    for col in range(2):
        
        axarr[row, col].plot(range(0 , len(y_trainR)), y_trainR[:,count], 'g', lw=1, label=f"R {count+1}")
        axarr[row, col].plot(range(0 , len(y_trainS)), y_trainS[:,count], 'b', lw=1, label=f"S {count+1}")
        axarr[row, col].plot(range(0 , len(y_trainBW)), y_trainBW[:,count], 'r', lw=1, label=f"BW {count+1}")
        
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1} error")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.yscale("log")
plt.ylim(bottom=10e-10)
plt.xlim(0, len(y_test))
plt.savefig(f"{plt_save}y_train modes.png")
"""