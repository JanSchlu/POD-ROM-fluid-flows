#!/usr/bin/env python3

import torch as pt
import matplotlib.pyplot as plt
from functions import *
from params import data_save, SVD_modes, p_steps, model_params

plt_save = "plot/"

lenTrain = pt.load(f"{data_save}lenTrain.pt")                                                       
lenTest = pt.load(f"{data_save}lenTest.pt")
test_data = pt.load(f"{data_save}test_data.pt")
y_test = pt.load(f"{data_save}y_test.pt")
window_times = pt.load(f"{data_save}window_times.pt")
train_loss = pt.load(f"{data_save}train_loss.pt")


# increase plot resolution
plt.rcParams["figure.dpi"] = 320

#######################################################################################
# load model
#######################################################################################

best_model = FirstNN(**model_params)
best_model.load_state_dict(pt.load(f"{data_save}best_model_train.pt"))

#######################################################################################
# predict next timestep
#######################################################################################

pred = pt.ones([lenTest-1-p_steps,SVD_modes])                                           # pred len(test_data)-1-p_steps
for i in range (0, len(pred)):
    pred[i] = best_model(test_data[i]).squeeze()
pred = pred.detach().numpy()

#######################################################################################
# sequential prediction
#######################################################################################

predSeq = pt.ones([lenTest-p_steps,SVD_modes+SVD_modes*p_steps])                 
predSeq_print = pt.ones([lenTest-1-p_steps,SVD_modes])
predSeq[0] = test_data[0]                                                               #start is last timestep of trainData
predSeq_print[0] = test_data[0, SVD_modes*p_steps:]
for i in range (1, len(predSeq)):
    prediction = best_model(predSeq[i-1]).squeeze()
    predSeq[i] = pt.cat((predSeq[i-1,SVD_modes:], prediction))
    predSeq_print[i-1] = prediction 
predSeq_print = predSeq_print.detach().numpy()
#######################################################################################
# plot training/validation loss
#######################################################################################

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(1, epochs+1), train_loss, lw=1.0, label="training loss")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.xlim(1, epochs+1)
plt.yscale("log")
plt.ylim(bottom=10e-7)
plt.legend()
plt.savefig(f"{plt_save}loss.png")

#######################################################################################
# plot prediciton mode
#######################################################################################

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(y_test)), y_test[:,count], 'b', lw=0.5, label=f"coeff. mode {i+1}")
        axarr[row, col].plot(range(0 , len(y_test)), pred[:,count], 'g--', lw=0.5, label=f"coeff. mode {i+1}")
        axarr[row, col].plot(range(0 , len(y_test)), predSeq_print[:,count], 'r--', lw=0.5, label=f"coeff. mode {i+1}")
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, len(y_test))
plt.ylim(bottom=10e-10)
plt.savefig(f"{plt_save}prediction.png")

#######################################################################################
# error plot
#######################################################################################

err = pt.zeros(len(test_data))
errSeq = int
fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
err = (y_test-pred)**2
errSeq = (y_test-predSeq_print)**2
meanErr = pt.sum(err,1)/SVD_modes
meanErrSeq = pt.sum(errSeq,1)/SVD_modes
err = err.detach().numpy()
errSeq = errSeq.detach().numpy()
meanErr = meanErr.detach().numpy()
meanErrSeq = meanErrSeq.detach().numpy()

for row in range(2):
    for col in range(2):
        #axarr[row, col].plot(times_num, modeCoeff[:,count], lw=1, label=f"coeff. mode {i+1}")
        axarr[row, col].plot(range(0 , len(y_test)), err[:,count], 'g', lw=1, label=f"coeff. mode {i+1}")
        axarr[row, col].plot(range(0 , len(y_test)), errSeq[:,count], 'r', lw=1, label=f"coeff. mode {i+1}")
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
plt.xlim(0, len(y_test))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError.png")
