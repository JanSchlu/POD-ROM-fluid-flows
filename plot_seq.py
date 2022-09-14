#!/usr/bin/env python3
'''
plot sequential


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

predS = pt.load(f"{data_save}predS.pt")
predR = pt.load(f"{data_save}predR.pt")
predBW = pt.load(f"{data_save}predBW.pt")
predR[0] = y_test_norm_R[0] 
MinInS = pt.load(f"{data_save}MinInS.pt")
MaxInS = pt.load(f"{data_save}MaxInS.pt")
modes = pt.load(f"{data_save}modeCoeffBinary.pt")


MinOutS = pt.load(f"{data_save}MinOutS.pt")
MinOutS = MinOutS[:SVD_modes]
MaxOutS = pt.load(f"{data_save}MaxOutS.pt")
MaxOutS = MaxOutS[:SVD_modes]
MinOutR = pt.load(f"{data_save}MinOutR.pt")
MinOutR = MinOutR[:SVD_modes]
MaxOutR = pt.load(f"{data_save}MaxOutR.pt")
MaxOutR = MaxOutR[:SVD_modes]
MinOutBW = pt.load(f"{data_save}MinOutBW.pt")
MinOutBW = MinOutBW[:SVD_modes]
MaxOutBW = pt.load(f"{data_save}MaxOutBW.pt")
MaxOutBW = MaxOutBW[:SVD_modes]


predS_period = pt.load(f"{data_save}predS_period.pt")
predR_period = pt.load(f"{data_save}predR_period.pt")
predBW_period = pt.load(f"{data_save}predBW_period.pt")


data2 = pt.load(f"{data_save}modeCoeffBinary.pt")
data2= data2[:,:SVD_modes]

test_data, y_test_orig = dataManipulator_yTest(data2, SVD_modes, p_steps)

#######################################################################################
# plot training/validation loss
#######################################################################################
""""
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
"""



#######################################################################################
# plot prediciton mode
#######################################################################################

y_testS = (y_test_norm_S[:-1] - MinOutS )  /(MaxOutS-MinOutS)

S = predS
R = predR
BW = pt.tensor(predBW)

y_testS = y_test_norm_S[:-4]
y_testR = y_test_norm_R[:-4]
y_testBW = y_test_norm_BW[:-1]


fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(y_testS)), y_testS[:,count], 'b', lw=0.5, label=f"orig")
        axarr[row, col].plot(range(0 , len(predS_period)), predS_period[:,count], 'r', lw=0.5, label=f"S_seq")
        axarr[row, col].plot(range(0 , len(predR_period)), predR_period[:,count], 'g', lw=0.5, label=f"R_seq")
        axarr[row, col].plot(range(0 , len(predBW_period)), predBW_period[:,count], lw=0.5, label=f"BW_seq")
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel("predicted timesteps")
plt.xlim(0, 700)
#plt.ylim(bottom=10e-10)
plt.legend()
plt.savefig(f"{plt_save}prediction_seq.png")

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

        axarr[row, col].plot(range(0 , len(test_data_norm_S)), test_data_norm_S[:,count], 'b', ls = '--',lw=0.5, label=f"Eingang x")
#        axarr[row, col].plot(range(0 , len(test_dataS)), test_dataS[:,count], ls = '--', lw=0.5, label=f"Eingang x")
        axarr[row, col].plot(range(0 , len(y_testS)), y_testS[:,count],  'g', ls = '--',lw=0.5, label=f"y_test S")
        axarr[row, col].plot(range(0 , len(predS_period)), predS_period[:,count],  'g', lw=0.5, label=f"S_period")       
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
plt.savefig(f"{plt_save}predictionS_period.png")
"""
fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
#        axarr[row, col].plot(range(0 , len(test_data_norm_R)), test_data_norm_R[:,count], 'b', ls = '--',lw=0.5, label=f"Eingang x")
#        axarr[row, col].plot(range(0 , len(y_test)), y_test[:,count],  'r', ls = '--',lw=0.5, label=f"y Vergleich S")
#        axarr[row, col].plot(range(0 , len(y_test)), predS[:,count],  'r', lw=0.5, label=f"S")       
        axarr[row, col].plot(range(0 , len(y_testR)), y_testR[:,count], 'g', ls = '--', lw=0.5, label=f"y_test R")  
        axarr[row, col].plot(range(0 , len(predR_period)), predR_period[:,count], 'g', lw=0.5, label=f"R_period")
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
plt.savefig(f"{plt_save}predictionR_period.png")
"""
fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(range(0 , len(test_data_norm_BW)), test_data_norm_BW[:,count], 'b', ls = '--',lw=0.5, label=f"Eingang x")
#        axarr[row, col].plot(range(0 , len(y_test)), y_test[:,count],  'r', ls = '--',lw=0.5, label=f"y Vergleich S")
#        axarr[row, col].plot(range(0 , len(y_test)), predS[:,count],  'r', lw=0.5, label=f"S")       
#        axarr[row, col].plot(range(0 , len(y_testR)), y_testR[:,count], 'g', ls = '--', lw=0.5, label=f"y Vergleich R")  
        axarr[row, col].plot(range(0 , len(predBW_period)), predBW_period[:,count], 'g', lw=0.5, label=f"BW_period")
        axarr[row, col].plot(range(0 , len(y_testBW)), y_testBW[:,count],'g', ls='--',lw=0.5, label=f"y_test BW")     
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
plt.savefig(f"{plt_save}predictionBW_period.png")

"""

err = pt.zeros(len(test_data_norm_S))
fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
#y_testS = (max_y_testS-min_y_testS)*y_testS + min_y_testS  # ist schon entnormeirt
#y_testR = (MaxOutR - MinOutR)*y_testR + MinOutR
#y_testBW = (MaxOutBW - MinOutBW)*y_testBW + MinOutBW

#err = (y_testS-predS)**2
errSeq = (y_testS-predS_period)**2
#meanErr = pt.sum(err,1)/SVD_modes
meanErrSeq = pt.sum(errSeq,1)/SVD_modes
#err = err.detach().numpy()
errSeq = errSeq.detach().numpy()
#meanErr = meanErr.detach().numpy()
meanErrSeq = meanErrSeq.detach().numpy()

#errR = (y_testR-predR)**2
errRSeq = (y_testR-predR_period)**2
#meanErrR = pt.sum(errR,1)/SVD_modes
meanErrRSeq = pt.sum(errRSeq,1)/SVD_modes
#errR = errR.detach().numpy()
errRSeq = errRSeq.detach().numpy()
#meanErrR = meanErrR.detach().numpy()
meanErrRSeq = meanErrRSeq.detach().numpy()

y_test_orig = y_test_orig[:-2]

#errBW = (y_testBW-predBW)**2
#errBWSeq = (y_testBW-predBW_period)**2
#meanErrbw = pt.sum(errBW,1)/SVD_modes
#meanErrbwSeq = pt.sum(errBWSeq,1)/SVD_modes
#errBW = errBW.detach().numpy()
#errBWSeq = errBWSeq.detach().numpy()
#meanErrbw = meanErrbw.detach().numpy()
#meanErrbwSeq = meanErrbwSeq.detach().numpy()

for row in range(2):
    for col in range(2):
#        axarr[row, col].plot(times_num, modeCoeff[:,count], lw=1, label=f"coeff. mode {i+1}")
#        axarr[row, col].plot(range(0 , len(y_testS)), err[:,count], 'g', lw=1, label=f"S" ,)
        axarr[row, col].plot(range(0 , len(y_testS)), errSeq[:,count], 'r', lw=1, label=f"S")

#        axarr[row, col].plot(range(0 , len(y_testR)), errR[:,count], 'g:', lw=1, label=f"R",)
        axarr[row, col].plot(range(0 , len(y_testR)), errRSeq[:,count], 'r:', lw=1, label=f"R")

#        axarr[row, col].plot(range(0 , len(y_testBW)), errBW[:,count], 'g--', lw=1, label=f"BW",)
#        axarr[row, col].plot(range(0 , len(y_testBW)), errBWSeq[:,count], 'r--', lw=1, label=f"BW")
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
"""
fig, ax = plt.subplots()
epochs = len(train_loss)                                      
#plt.plot(range(0, len(meanErr)), meanErr, 'g', lw=1.0, label="prediction error S")
plt.plot(range(0, len(y_testS)), meanErrSeq, 'r', lw=1.0, label= "seq. prediction error S")
plt.xlim(0, len(meanErrSeq))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError_S.png")
"""
fig, ax = plt.subplots()
epochs = len(train_loss)                                      
#plt.plot(range(0, len(meanErrR)), meanErrR, 'g', lw=1.0, label="prediction error R")
plt.plot(range(0, len(y_testR)), meanErrRSeq, 'r', lw=1.0, label= "seq. prediction error R")
plt.xlim(0, len(meanErrRSeq))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError_res.png")
"""
fig, ax = plt.subplots()
epochs = len(train_loss)                                      
#plt.plot(range(0, len(meanErrbw)), meanErrbw, 'g', lw=1.0, label="prediction error BW")
plt.plot(range(0, len(y_testBW)), meanErrbwSeq, 'r', lw=1.0, label= "seq. prediction error BW")
plt.xlim(0, len(meanErrbwSeq))
plt.xlabel("preditected timesteps")
plt.ylabel("mean error")
plt.yscale("log")
plt.legend()
plt.savefig(f"{plt_save}meanError_bw.png")
"""