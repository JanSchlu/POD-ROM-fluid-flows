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
y_test_norm_S = pt.load(f"{data_save}y_test_norm_S.pt")
y_test_norm_R = pt.load(f"{data_save}y_test_norm_R.pt")
y_test_norm_BW = pt.load(f"{data_save}y_test_norm_BW.pt") 
test_data_norm_S = pt.load(f"{data_save}test_data_norm_S.pt")
test_data_norm_R = pt.load(f"{data_save}test_data_norm_R.pt")
test_data_norm_BW = pt.load(f"{data_save}test_data_norm_BW.pt")
y_train_norm_S = pt.load(f"{data_save}y_train_norm_S.pt")
train_data_norm_S = pt.load(f"{data_save}train_data_norm_S.pt")
train_data_norm_R = pt.load(f"{data_save}train_data_norm_R.pt")
train_data_norm_BW = pt.load(f"{data_save}train_data_norm_BW.pt")
y_train_norm_R = pt.load(f"{data_save}y_train_norm_R.pt")
y_train_norm_BW = pt.load(f"{data_save}y_train_norm_BW.pt")
train_loss = pt.load(f"{data_save}train_loss.pt")
train_lossR = pt.load(f"{data_save}train_lossR.pt")
train_lossBW = pt.load(f"{data_save}train_lossBW.pt")



scalerdict = dataloader(f"{data_save}scalerdict.pkl")
predS = pt.load(f"{data_save}predS.pt")
predR = pt.load(f"{data_save}predR.pt")
predBW = pt.load(f"{data_save}predBW.pt")
predR[0] = y_test_norm_R[0] 
modes = pt.load(f"{data_save}modeCoeffBinary.pt")

#predS_period = pt.load(f"{data_save}predS_period.pt")
#predR_period = pt.load(f"{data_save}predR_period.pt")
#predBW_period = pt.load(f"{data_save}predBW_period.pt")


data2 = pt.load(f"{data_save}modeCoeffBinary.pt")
data2= data2[:,:SVD_modes]

test_data, y_test_orig = dataManipulator_yTest(data2, SVD_modes, p_steps)

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
# plot data
#######################################################################################

fig, axarr = plt.subplots(5, 2, sharex=True, sharey=True)
count = 0
for row in range(5):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(train_data_norm_S)), train_data_norm_S[:,count],         'b',    lw=0.5, label=f"S train x")
        axarr[row, col].plot(range(0 , len(y_train_norm_S)), y_train_norm_S[:,count],    'b',   ls='--',     lw=0.5, label=f"S train y")
        axarr[row, col].plot(range(0 , len(test_data_norm_S)), test_data_norm_S[:,count],           'r',    lw=0.5, label=f"S test x")
        axarr[row, col].plot(range(0 , len(y_test_norm_S)), y_test_norm_S[:,count],      'r',   ls='--',        lw=0.5, label=f"S test y")
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, 100)
#plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}DataS.png")


fig, axarr = plt.subplots(5, 2, sharex=True, sharey=True)
count = 0
for row in range(5):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(train_data_norm_R)), train_data_norm_R[:,count], 'b', lw=0.5, label=f"train R x")
        axarr[row, col].plot(range(0 , len(y_train_norm_R)), y_train_norm_R[:,count], 'b', ls='--', lw=0.5, label=f"train R y")
        axarr[row, col].plot(range(0 , len(test_data_norm_R)), test_data_norm_R[:,count],'r', lw=0.5, label=f"test R x")
        axarr[row, col].plot(range(0 , len(y_test_norm_R)), y_test_norm_R[:,count],'r',   ls='--',  lw=0.5, label=f"test R y")
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, 100)
#plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}DataR.png")


fig, axarr = plt.subplots(5, 2, sharex=True, sharey=True)
count = 0
for row in range(5):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(train_data_norm_BW)), train_data_norm_BW[:,count], 'b', lw=0.5, label=f"train BW x")
        axarr[row, col].plot(range(0 , len(y_train_norm_BW)), y_train_norm_BW[:,count], 'b',   ls='--', lw=0.5, label=f"train BW y")
        axarr[row, col].plot(range(0 , len(test_data_norm_BW)), test_data_norm_BW[:,count],'r', lw=0.5, label=f"test BW x")
        axarr[row, col].plot(range(0 , len(y_test_norm_BW)), y_test_norm_BW[:,count],'r',   ls='--',lw=0.5, label=f"test BW y")
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, 100)
#plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}DataBW.png")


#######################################################################################
# plot prediciton mode
#######################################################################################

y_testS = (y_test_norm_S[:-1] - scalerdict["MinOutS"] )  /(scalerdict["MaxOutS"]-scalerdict["MinOutS"])

S = predS
R = predR                                        
BW = pt.tensor(predBW)

y_testS = y_test_norm_S[:-p_steps-1]
y_testR = y_test_norm_R[:-p_steps-1]
y_testBW = y_test_norm_BW[:-p_steps-1]

"""
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
"""
#######################################################################################
# error plot
#######################################################################################




#######################################################################################
# plot prediciton mode
#######################################################################################
#predS = (- min_y_testS + predS) / (max_y_testS-min_y_testS)
#predR2 = (-min_y_testR+predR )/(max_y_testR-min_y_testR)
#predBW = (pt.tensor(predBW)- MinOutBW)  /(MaxOutBW-MinOutBW)
#y_testS = (max_y_testS-min_y_testS)*y_testS + min_y_testS
#y_testR = (max_y_testR-min_y_testR)*y_testR + min_y_testR
#y_testBW = (max_y_testBW-min_y_testBW)*y_testBW + min_y_testBW
#test_dataS = (MaxInDataS - MinInDataS)*test_dataS + MinInDataS
"""
fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
#        axarr[row, col].plot(range(0 , len(test_dataS)), test_dataS[:,count], ls = '--', lw=0.5, label=f"Eingang x")
        axarr[row, col].plot(range(0 , len(y_testS)), y_testS[:,count],  'r', ls = '--',lw=0.5, label=f"y Vergleich S")
        axarr[row, col].plot(range(0 , len(y_testS)), predS[:,count],  'r', lw=0.5, label=f"S")       
#        axarr[row, col].plot(range(0 , len(y_testR)), y_testR[:,count], 'g', ls = '--', lw=0.5, label=f"y Vergleich R")  
 #       axarr[row, col].plot(range(0 , len(y_test)), predR[:,count], 'g', lw=0.5, label=f"R")
  #      axarr[row, col].plot(range(0 , len(y_testBW)), y_testBW[:,count],'b', ls='--',lw=0.5, label=f"y Vergleich BW")     
   #     axarr[row, col].plot(range(0 , len(y_test)), predBW[:,count],'b', lw=0.5, label=f"BW")     
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, 100)
#plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}predictionS.png")
"""
fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(test_data_norm_S)), test_data_norm_S[:,count], 'r', ls = '--',lw=0.5, label=f"Eingang x")
#        axarr[row, col].plot(range(0 , len(y_test)), y_test[:,count],  'r', ls = '--',lw=0.5, label=f"y Vergleich S")
#        axarr[row, col].plot(range(0 , len(y_test)), predS[:,count],  'r', lw=0.5, label=f"S")       
        axarr[row, col].plot(range(0 , len(y_testR)), y_testR[:,count], 'g', ls = '--', lw=0.5, label=f"y Vergleich R")  
        axarr[row, col].plot(range(0 , len(predR)), predR[:,count], 'g', lw=0.5, label=f"R")
  #      axarr[row, col].plot(range(0 , len(y_testBW)), y_testBW[:,count],'b', ls='--',lw=0.5, label=f"y Vergleich BW")     
   #     axarr[row, col].plot(range(0 , len(y_test)), predBW[:,count],'b', lw=0.5, label=f"BW")     
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, 100)
#plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}predictionR.png")

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(test_data_norm_BW)), test_data_norm_BW[:,count], 'r', ls = '--',lw=0.5, label=f"Eingang x")
#        axarr[row, col].plot(range(0 , len(y_test)), y_test[:,count],  'r', ls = '--',lw=0.5, label=f"y Vergleich S")
#        axarr[row, col].plot(range(0 , len(y_test)), predS[:,count],  'r', lw=0.5, label=f"S")       
#        axarr[row, col].plot(range(0 , len(y_testR)), y_testR[:,count], 'g', ls = '--', lw=0.5, label=f"y Vergleich R")  
        axarr[row, col].plot(range(0 , len(predBW)), predBW[:,count], 'g', lw=0.5, label=f"BW")
        axarr[row, col].plot(range(0 , len(y_testBW)), y_testBW[:,count],'b', ls='--',lw=0.5, label=f"y Vergleich BW")     
#        axarr[row, col].plot(range(0 , len(y_testBW)), predBW[:,count],'b', lw=0.5, label=f"BW")     
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, 100)
#plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}predictionBW.png")



err = pt.zeros(len(test_data_norm_S))
fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
#y_testS = (max_y_testS-min_y_testS)*y_testS + min_y_testS  # ist schon entnormeirt
#y_testR = (MaxOutR - MinOutR)*y_testR + MinOutR
#y_testBW = (MaxOutBW - MinOutBW)*y_testBW + MinOutBW

err = (y_testS-predS)**2
#errSeq = (y_test-predS_period)**2
meanErr = pt.sum(err,1)/SVD_modes
#meanErrSeq = pt.sum(errSeq,1)/SVD_modes
err = err.detach().numpy()
#errSeq = errSeq.detach().numpy()
meanErr = meanErr.detach().numpy()
#meanErrSeq = meanErrSeq.detach().numpy()

errR = (y_testR-predR)**2
#errRSeq = (y_test-predR_period)**2
meanErrR = pt.sum(errR,1)/SVD_modes
#meanErrRSeq = pt.sum(errRSeq,1)/SVD_modes
errR = errR.detach().numpy()
#errRSeq = errRSeq.detach().numpy()
meanErrR = meanErrR.detach().numpy()
#meanErrRSeq = meanErrRSeq.detach().numpy()
y_test_orig = y_test_orig[:-2]
errBW = (y_testBW-predBW)**2
#errBWSeq = (y_test-predBW_period)**2
meanErrbw = pt.sum(errBW,1)/SVD_modes
#meanErrbwSeq = pt.sum(errBWSeq,1)/SVD_modes
errBW = errBW.detach().numpy()
#errBWSeq = errBWSeq.detach().numpy()
meanErrbw = meanErrbw.detach().numpy()
#meanErrbwSeq = meanErrbwSeq.detach().numpy()

for row in range(2):
    for col in range(2):
#        axarr[row, col].plot(times_num, modeCoeff[:,count], lw=1, label=f"coeff. mode {i+1}")
        axarr[row, col].plot(range(0 , len(y_testS)), err[:,count], 'g', lw=1, label=f"S" ,)
#        axarr[row, col].plot(range(0 , len(y_test)), errSeq[:,count], 'r', lw=1, label=f"coeff. mode {count+1}")

        axarr[row, col].plot(range(0 , len(y_testR)), errR[:,count], 'g:', lw=1, label=f"R",)
#        axarr[row, col].plot(range(0 , len(y_test)), errRSeq[:,count], 'r:', lw=1, label=f"coeff. mode {count+1}")

        axarr[row, col].plot(range(0 , len(y_testBW)), errBW[:,count], 'g--', lw=1, label=f"BW",)
#        axarr[row, col].plot(range(0 , len(y_test)), errBWSeq[:,count], 'r--', lw=1, label=f"coeff. mode {count+1}")
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1} error")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.yscale("log")
plt.ylim(bottom=10e-10)
plt.xlim(0, len(err))
plt.legend()
plt.savefig(f"{plt_save}error.png")

#######################################################################################
# plot mean error plot
#######################################################################################

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(0, len(meanErr)), meanErr, 'g', lw=1.0, label="prediction error S")
#plt.plot(range(0, len(y_test)), meanErrSeq, 'r', lw=1.0, label= "seq. prediction error S")
plt.xlim(0, len(meanErr))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError_S.png")

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(0, len(meanErrR)), meanErrR, 'g', lw=1.0, label="prediction error R")
#plt.plot(range(0, len(y_test)), meanErrRSeq, 'r', lw=1.0, label= "seq. prediction error R")
plt.xlim(0, len(meanErrR))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError_res.png")

fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(0, len(meanErrbw)), meanErrbw, 'g', lw=1.0, label="prediction error BW")
#plt.plot(range(0, len(y_test)), meanErrbwSeq, 'r', lw=1.0, label= "seq. prediction error BW")
plt.xlim(0, len(meanErrbw))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError_bw.png")
