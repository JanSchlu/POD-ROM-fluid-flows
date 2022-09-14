#!/usr/bin/env python3
"""
code for SVD, uses data from flowtorch -> of_cylinder2d_binary

SVD for all Re numbers with all timesteps
plot modes over time

first test
"""

import torch as pt
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis import SVD
import sys
from matplotlib import pyplot as plt

plt_save = "/home/jan/POD-ROM-fluid-flows/run/plot/"
from params import data_save
sys.path.append('/home/jan/POD-ROM-fluid-flows/')
def modeMaker(path):
    loader = FOAMDataloader(path)
    times = loader.write_times
    fields = loader.field_names
    print(f"Number of available snapshots: {len(times)}")
    # load vertices and discard z-coordinate
    vertices = loader.vertices[:, :2]
    mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])
    window_times = [time for time in times if float(time) >= 1.0]                                       #starts from 0 -> all timesteps
    data_matrix = pt.zeros((mask.sum().item(), len(window_times)), dtype=pt.float32)
    for i, time in enumerate(window_times):
       # load the vorticity vector field, take the z-component [:, 2], and apply the mask
        data_matrix[:, i] = pt.masked_select(loader.load_snapshot("vorticity", time)[:, 2], mask)
    # subtract the temporal mean
    data_matrix -= pt.mean(data_matrix, dim=1).unsqueeze(-1)
    svd = SVD(data_matrix)
    modeCoeff = pt.zeros(1)
    modeCoeff = svd.V#*svd.s    # Mode coefficients
    pt.save(window_times,f"{data_save}window_times.pt")
    return modeCoeff, window_times


pat = "/home/jan/POD-ROM-fluid-flows/run/Re56"
Re56, window_times = modeMaker(pat)

times_num = [float(time) for time in window_times]

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(times_num, Re56[:, count], lw=1, label=f"coeff. mode {count+1}")
        axarr[row, col].set_xlim(min(times_num), max(times_num))
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel(r"$t$ in $s$")
plt.savefig(f"{plt_save}Re56modes.png")


pat = "/home/jan/POD-ROM-fluid-flows/run/Re142"
Re142, window_times = modeMaker(pat)

times_num = [float(time) for time in window_times]

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(times_num, Re142[:, count], lw=1, label=f"coeff. mode {count+1}")
        axarr[row, col].set_xlim(min(times_num), max(times_num))
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel(r"$t$ in $s$")
plt.savefig(f"{plt_save}Re142modes.png")


pat = "/home/jan/POD-ROM-fluid-flows/run/Re198"
Re198, window_times = modeMaker(pat)

times_num = [float(time) for time in window_times]

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(times_num, Re198[:, count], lw=1, label=f"coeff. mode {count+1}")
        axarr[row, col].set_xlim(min(times_num), max(times_num))
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel(r"$t$ in $s$")
plt.savefig(f"{plt_save}Re198modes.png")


pat = "/home/jan/POD-ROM-fluid-flows/run/Re302"
Re302, window_times = modeMaker(pat)

times_num = [float(time) for time in window_times]

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(times_num, Re302[:, count], lw=1, label=f"coeff. mode {count+1}")
        axarr[row, col].set_xlim(min(times_num), max(times_num))
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel(r"$t$ in $s$")
plt.savefig(f"{plt_save}Re302modes.png")



pat = "/home/jan/POD-ROM-fluid-flows/run/Re392"
Re392, window_times = modeMaker(pat)


## plot modes over time


times_num = [float(time) for time in window_times]

fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(times_num, Re392[:, count], lw=1, label=f"coeff. mode {count+1}")
        axarr[row, col].set_xlim(min(times_num), max(times_num))
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel(r"$t$ in $s$")
plt.savefig(f"{plt_save}Re392modes.png")
