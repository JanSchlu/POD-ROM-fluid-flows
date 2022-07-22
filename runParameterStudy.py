#!/usr/bin/env python3


import torch as pt
import os
from utils_parameterStudy import lhs_sampling, change_line, myround
#################################################################
# latin hypercube sampling for Reynoldsnumber

N_s =5
x_min = [50]
x_max = [400]
Re_samples = lhs_sampling(x_min, x_max, N_s)
print(Re_samples)

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

    os.system ("cp -r test_cases/cylinder2D_base/ run/Re" + f'{int(Re_samples[0][entry])}' + "/")
    filepath="run/Re" + f'{int(Re_samples[0][entry])}' + "/system/setExprBoundaryFieldsDict"
    line = "            expression #{ 4*" + f'{u_max[entry][0]}'[:5] + "*pos().y()*(0.41-pos().y())/(0.41*0.41)*$[(vector)vel.dir] #};\n"
    change_line(filepath, line, 28)

#### -> Zeilenänderung des Zeitschrittes für jede Re in /system/controlDict


    filepath="run/Re" + f'{int(Re_samples[0][entry])}' + "/system/controlDict"
    CFL = 0.6
    velocity = float(u_max[entry][0])
    deltaT = myround((CFL* 59.2)/float(u_max[entry][0]),1)

    print(deltaT, u_max[entry][0])
    line = "deltaT          " + f'{deltaT}' + "e-05;\n"
    change_line(filepath, line, 27)

#### -> Zeilenänderung des write Intervall für jede Re in /system/controlDict
    #writeInterval_target = 5e-3
    #writeInterval = writeInterval_target/deltaT
    writeInterval = 1e-2
    print("writeInterval",u_max[entry][0],": ",writeInterval )
    line = "writeInterval   " + f'{writeInterval}'[:5] + ";\n"
    #change_line(filepath, line, 31)


#################################################################
#simulate all cases

for Re in range(0,len(Re_samples[0])):
    os.system('run/Re'+f'{int(Re_samples[0][Re])}'+'/Allrun.singularity')


# write Intervall in controlDict -> 5e-4 (= timestep for Re = 50)
# deltaT unter Berücksichtigung von Re (deltaT_min = 6.25e-5, deltaT_max = 5e-4)
## zeitschritte manchmal in e schreibweise
## change simulated time and timestepsize in /system/controlDict
# deltaT_max = 0.006 for Re400 -> CFL=1 
# CFL Zahl konst halten, Zeitschritt an Re anpassen
# Ausgabe Zeitschritt konstant halten
# Orientierung an Fabians Bachelor Arbeit für die Frequenz/Strohalzahl über Re
# OpenFoam Version for /cylinder2D_base is v1912 -> used version is v2106 -> require some changes -> wrong version used, use singularity

## Wie viele Pkte pro Schwingung braucht das Netz zur korrekten Wiedergabe des zeitlichen Verlaufs?
## Wie viele zeitliche Datenpunkte braucht das Netz fürs Training?