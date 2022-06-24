#!/usr/bin/env python3

import torch as pt

#################################################################
# latin hypercube sampling

def lhs_sampling(x_min, x_max, n_samples):
    assert len(x_min) == len(x_max)
    n_parameters = len(x_min)
    samples = pt.zeros((n_parameters, n_samples))
    for i, (lower, upper) in enumerate(zip(x_min, x_max)):
        bounds = pt.linspace(lower, upper, n_samples+1)
        rand = bounds[:-1] + pt.rand(n_samples) * (bounds[1:]-bounds[:-1])
        samples[i, :] = rand[pt.randperm(n_samples)]
    return samples


N_s = 10
x_min = [5]
x_max = [400]
Re_samples = lhs_sampling(x_min, x_max, N_s)

#################################################################
# calculate velocity

nu = 1e-3
d = 0.1
u_infty = pt.zeros((len(Re_samples[0]),1))
u_max = pt.zeros((len(Re_samples[0]),1))
for i in range(len(Re_samples[0])):
    u_infty[i] =(Re_samples[0][i] * nu)/d 
    u_max[i] = round(1.5 * u_infty[i].item(),2)

#################################################################
# change line in setExprBoundaryFieldsDict

filepath="cylinder2D_base/system/setExprBoundaryFieldsDict"

def change_line(path, velocity):
    setBoundary_file = open(path, "r")
    lines = setBoundary_file.readlines()
    lines[28] = "            expression #{ 4*" + f'{velocity}'[:5] + "*pos().y()*(0.41-pos().y())/(0.41*0.41)*$[(vector)vel.dir] #};\n"
    setBoundary_file = open(path, "w")
    setBoundary_file.writelines(lines)
    setBoundary_file.close()

change_line(filepath, u_max[0][0])

#################################################################
#   1.  create new dict for each Re
#   2.  simulate first Re
#   3.  store TimeDirectories in Re dict
#   4.  simulate new Re
#   repeat 3. and 4. for all Re

#   change simulated time and timestepsize in /system/controlDict
#   OpenFoam Version for /cylinder2D_base is v1912 -> used version is v2106 -> require some changes