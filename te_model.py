#!/usr/bin/env python3
"""
code for testing the neural network and functions
"""
from matplotlib import pyplot as plt
import torch as pt
from functions import *
from params import SVD_modes
from test_functions import predictor_backward, predictor_residual, predictor_sequential, predictor_residual_period
train_loss = pt.zeros(10000)

plt_save = "/home/jan/POD-ROM-fluid-flows/run/plot/"
#######################################################################################
# Initialize model
#######################################################################################

model_params = {
"n_inputs": 1,
"n_outputs": 1,
"n_layers": 0,
"n_neurons": 1,
"activation": pt.nn.Identity()	#fast and accurate
}


data = pt.zeros(100,1)
for i in range (1,100):
    data[i]=data[i-1]+0.01
train_data, y_train = rearrange_data(data,0)
train_data = train_data[:,]
test_data, y_test = rearrange_data(data,0)
y_test = y_test[:-1]
y_trainR = recalculate_output(train_data,y_train,"residual",1)
y_trainBW = recalculate_output(train_data,y_train,"backward",1)
y_train = recalculate_output(train_data,y_train,"sequential",1)
print(y_train, y_trainR, y_trainBW)
train_data=train_data[:-1]

predS = predictor_sequential(test_data)    
predR = predictor_residual(test_data)
predBW = predictor_backward(test_data)
predR_period = predictor_residual_period(test_data)
#######################################################################################
# error plot
#######################################################################################

err = pt.zeros(len(y_test))
fig, ax = plt.subplots()

err = (y_test-predS)**2
meanErr = pt.sum(err,1)/SVD_modes 
meanErr = meanErr.detach().numpy()

errR = (y_test-predR)**2
meanErrR = pt.sum(errR,1)/SVD_modes 
meanErrR = meanErrR.detach().numpy()

errBW = (y_test-predBW)**2
meanErrbw = pt.sum(errBW,1)/SVD_modes 
meanErrbw = meanErrbw.detach().numpy()

errR_period = (y_test-predR_period)**2
meanErrR_period = pt.sum(errR_period,1)/SVD_modes 
meanErrR_period = meanErrR_period.detach().numpy()
#######################################################################################
# plot mean error plot
#######################################################################################

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(0, len(y_test)), meanErr, 'g', lw=1.0, label="prediction error S")
plt.plot(range(0, len(y_test)), meanErrR, 'g:', lw=1.0, label="prediction error R")
plt.plot(range(0, len(y_test)), meanErrbw, 'g--', lw=1.0, label="prediction error BW")

plt.plot(range(0, len(y_test)), meanErrR_period, 'r--', lw=1.0, label="prediction error R_period")
plt.xlim(0, len(y_test))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
#plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanErrortest.png")


#######################################################################################
# plot prediciton mode
#######################################################################################

fig, ax = plt.subplots()
plt.plot(range(0 , len(y_test)), y_test[:], 'b--', lw=0.5, label="original")
plt.plot(range(0 , len(y_test)), predS[:], marker="1", lw=0.5, label="seqientiell")
plt.plot(range(0 , len(y_test)), predR[:], marker="+", lw=0.5, label="residuell")
plt.plot(range(0 , len(y_test)), predBW[:],marker="_", lw=0.5, label="backward")
plt.xlim(0, len(y_test))
plt.legend()
plt.savefig(f"{plt_save}predictiontest.png")