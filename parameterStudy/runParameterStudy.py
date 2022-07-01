#!/usr/bin/env python3


import torch as pt
import os
from utils import lhs_sampling, change_line
#################################################################
# latin hypercube sampling for Reynoldsnumber

N_s = 10
x_min = [5]
x_max = [400]
Re_samples = lhs_sampling(x_min, x_max, N_s)

#################################################################
# calculate velocity from Reynoldsnumber

nu = 1e-3
d = 0.1
u_infty = pt.zeros((len(Re_samples[0]),1))
u_max = pt.zeros((len(Re_samples[0]),1))
for i in range(len(Re_samples[0])):
    u_infty[i] =(Re_samples[0][i] * nu)/d 
    u_max[i] = round(1.5 * u_infty[i].item(),2)

#################################################################
# copy base case and change velocity line in setExprBoundaryFieldsDict

os.system ('test_cases/cylinder2D_base/Allclean')
for entry in range(0,len(Re_samples[0])):

    os.system ("cp -r test_cases/cylinder2D_base/ ../run/Re" + f'{int(Re_samples[0][entry])}' + "/")
    filepath="../run/Re" + f'{int(Re_samples[0][entry])}' + "/system/setExprBoundaryFieldsDict"
    change_line(filepath, u_max[entry][0])

#################################################################
#simulate all cases

for Re in range(len(Re_samples)):
    print(Re)
    os.system('../run/Re'+f'{int(Re_samples[0][Re])}'+'/Allrun.singularity')


## ausführung im run ordner
## singularity image globaly not available
## functions file globaly not available
## change simulated time and timestepsize in /system/controlDict
# deltaT_max = 0.006 for Re400 -> CFL=1 
# CFL Zahl konst halten, Zeitschritt an Re anpassen
# Ausgabe Zeitschritt konstant halten
# Orientierung an Fabians Bachelor Arbeit für die Frequenz/Strohalzahl über Re
# OpenFoam Version for /cylinder2D_base is v1912 -> used version is v2106 -> require some changes -> wrong version used, use singularity

## Wie viele Pkte pro Schwingung braucht das Netz zur korrekten Wiedergabe des zeitlichen Verlaufs?
## Wie viele zeitliche Datenpunkte braucht das Netz fürs Training?