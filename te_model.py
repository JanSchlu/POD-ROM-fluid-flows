#!/usr/bin/env python3
"""
code for testing the neural network and functions
"""
from matplotlib import pyplot as plt
import torch as pt
from functions import *
from params import SVD_modes

train_loss = pt.zeros(10000)

plt_save = "plot/"
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
y_trainR = recalculate_output(train_data,y_train,"residual")
y_trainBW = recalculate_output(train_data,y_train,"backward")
y_train = recalculate_output(train_data,y_train,"sequential")
train_data=train_data[:-1]


modelS = testNN()
#modelR = testNN(y_trainR)
#modelBW = testNN(y_trainBW)


p_steps =0
SVD_modes = 1
def predictor(data, prediction_update):
    predict = pt.ones([len(test_data)-1-p_steps,SVD_modes])                                           # pred len(test_data)-1-p_steps
    for i in range (0, len(predict)):
        if prediction_update == "sequential":
            predict[i] = data[i+1]
        if prediction_update == "residual":
            predict[0]= data[1]
            if i>0:
                predict[i] = data[i] + 0.01#modelR(data[i])
        if prediction_update == "backward":
            predict[0]=data[1]
            predict[1]=data[2]
            if i>1:
                predict[i] = 4/3*data[i] - 1/3*data[i-1] + 2/3*0.01*5e-3
    predict = predict.detach().numpy()
    return predict

predS = predictor(test_data, "sequential")    
predR = predictor(test_data, "residual")
predBW = predictor(test_data, "backward")
print("predR",predR)

#######################################################################################
# error plot
#######################################################################################

err = pt.zeros(len(y_test))
fig, ax = plt.subplots()

err = (y_test-predS)#**2
meanErr = pt.sum(err,1)/SVD_modes
err = err.detach().numpy()
meanErr = meanErr.detach().numpy()

errR = (y_test-predR)#**2
meanErrR = pt.sum(errR,1)/SVD_modes
errR = errR.detach().numpy()
meanErrR = meanErrR.detach().numpy()

errBW = (y_test-predBW)#**2
meanErrbw = pt.sum(errBW,1)/SVD_modes
errBW = errBW.detach().numpy()
meanErrbw = meanErrbw.detach().numpy()


plt.plot(range(0 , len(y_test)), err[:], 'g', lw=1, label="MSE Error S")
plt.plot(range(0 , len(y_test)), errR[:], 'g:', lw=1, label="MSE Error R")
plt.plot(range(0 , len(y_test)), errBW[:], 'g--', lw=1, label="MSE Error BW")

#plt.yscale("log")
#plt.ylim(bottom=10e-10)
plt.xlim(0, len(y_test))
plt.legend()
plt.savefig(f"{plt_save}errortest.png")

#######################################################################################
# plot mean error plot
#######################################################################################

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(0, len(y_test)), meanErr, 'g', lw=1.0, label="prediction error S")
plt.plot(range(0, len(y_test)), meanErrR, 'g:', lw=1.0, label="prediction error R")
plt.plot(range(0, len(y_test)), meanErrbw, 'g--', lw=1.0, label="prediction error BW")
plt.xlim(0, len(y_test))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
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