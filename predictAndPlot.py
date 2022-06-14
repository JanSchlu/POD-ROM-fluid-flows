#!/usr/bin/env python3
'''
code for prediction and plot

'''
from turtle import backward
import torch as pt
import matplotlib.pyplot as plt
from functions import *
from params import data_save, SVD_modes, p_steps, model_params

plt_save = "plot/"
lenTrain = pt.load(f"{data_save}lenTrain.pt")                                                       
lenTest = pt.load(f"{data_save}lenTest.pt")
test_data = pt.load(f"{data_save}test_data.pt")
y_test = pt.load(f"{data_save}y_test.pt")
#y_testR = pt.load(f"{data_save}y_testBW.pt")
#y_testBW = pt.load(f"{data_save}y_testBW.pt")
window_times = pt.load(f"{data_save}window_times.pt")
train_loss = pt.load(f"{data_save}train_loss.pt")
train_lossR = pt.load(f"{data_save}train_lossR.pt")
train_lossBW = pt.load(f"{data_save}train_lossBW.pt")
# increase plot resolution
plt.rcParams["figure.dpi"] = 320

#######################################################################################
# load model
#######################################################################################

best_model = FirstNN(**model_params)
best_model.load_state_dict(pt.load(f"{data_save}best_model_train.pt"))
data_saveR = "data/R/"
best_modelR = FirstNN(**model_params)
best_modelR.load_state_dict(pt.load(f"{data_saveR}best_model_train.pt"))
data_saveBW = "data/BW/"
best_modelBW = FirstNN(**model_params)
best_modelBW.load_state_dict(pt.load(f"{data_saveBW}best_model_train.pt"))

#######################################################################################
# one timestep prediction
#######################################################################################
def predictor(data, prediction_update):
    predict = pt.ones([lenTest-1-p_steps,SVD_modes])                                           # pred len(test_data)-1-p_steps
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
    predStore = pt.ones([lenTest-p_steps,SVD_modes+SVD_modes*p_steps])                 
    predicted = pt.ones([lenTest-1-p_steps,SVD_modes])
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

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(y_test)), y_test[:,count], 'b', lw=0.5, label=f"coeff. mode {count+1}")
        axarr[row, col].plot(range(0 , len(y_test)), predS[:,count], 'g--', lw=0.5, label=f"coeff. mode {count+1}")
        axarr[row, col].plot(range(0 , len(y_test)), predS_period[:,count], 'r--', lw=0.5, label=f"coeff. mode {count+1}")
        axarr[row, col].plot(range(0 , len(y_test)), predR[:,count], 'g:', lw=0.5, label=f"coeff. mode {count+1}")
        axarr[row, col].plot(range(0 , len(y_test)), predR_period[:,count], 'r:', lw=0.5, label=f"coeff. mode {count+1}")
        axarr[row, col].plot(range(0 , len(y_test)), predBW[:,count], 'g:', lw=0.5, label=f"coeff. mode {count+1}")
        axarr[row, col].plot(range(0 , len(y_test)), predBW_period[:,count], 'r:', lw=0.5, label=f"coeff. mode {count+1}")
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, len(y_test))
#plt.ylim(bottom=10e-10)
plt.savefig(f"{plt_save}prediction.png")

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
plt.plot(range(0, len(y_test)), meanErr, 'g', lw=1.0, label="prediction error")
plt.plot(range(0, len(y_test)), meanErrSeq, 'r', lw=1.0, label= "seq. prediction error")

plt.plot(range(0, len(y_test)), meanErrR, 'g:', lw=1.0, label="prediction error R")
plt.plot(range(0, len(y_test)), meanErrRSeq, 'r:', lw=1.0, label= "seq. prediction error R")

plt.plot(range(0, len(y_test)), meanErrbw, 'g--', lw=1.0, label="prediction error BW")
plt.plot(range(0, len(y_test)), meanErrbwSeq, 'r--', lw=1.0, label= "seq. prediction error BW")
plt.xlim(0, len(y_test))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError.png")
