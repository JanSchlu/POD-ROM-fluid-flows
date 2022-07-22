#!/usr/bin/env python3
"""
code for SVD, uses data from flowtorch -> of_cylinder2d_binary
"""
import torch as pt
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis import SVD

import sys
sys.path.append('/home/jan/POD-ROM-fluid-flows/')
from params import data_save

path = "/home/jan/POD-ROM-fluid-flows/run/of_cylinder2D_binary"           #path SVD

loader = FOAMDataloader(path)
times = loader.write_times
fields = loader.field_names
print(f"Number of available snapshots: {len(times)}")
print("First five write times: ", times[:5])
print(f"Fields available at t={times[-1]}: ", fields[times[-1]])

# load vertices and discard z-coordinate
vertices = loader.vertices[:, :2]
mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])
print(f"Selected vertices: {mask.sum().item()}/{mask.shape[0]}")

window_times = [time for time in times if float(time) >= 4.0]
data_matrix = pt.zeros((mask.sum().item(), len(window_times)), dtype=pt.float32)
for i, time in enumerate(window_times):
    # load the vorticity vector field, take the z-component [:, 2], and apply the mask
    data_matrix[:, i] = pt.masked_select(loader.load_snapshot("vorticity", time)[:, 2], mask)

# subtract the temporal mean
data_matrix -= pt.mean(data_matrix, dim=1).unsqueeze(-1)

svd = SVD(data_matrix)

modeCoeff = pt.zeros(1)
modeCoeff = svd.V#*svd.s    # Mode coefficients

print ("modeCoeff",modeCoeff.shape)
pt.save(svd.s,f"{data_save}svds.pt")
pt.save(modeCoeff,f"{data_save}modeCoeff.pt")
pt.save(window_times,f"{data_save}window_times.pt")