#!/usr/bin/env python3.8
"""
code for SVD, uses data from flowtorch
SVD for multiple datasets + Re number in dataset integrated
"""
import torch as pt
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis import SVD
import sys
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
    window_times = [time for time in times if float(time) >= 4.0]                               ### Strömung bei allen Re stationär?
    data_matrix = pt.zeros((mask.sum().item(), len(window_times)), dtype=pt.float32)
    for i, time in enumerate(window_times):
       # load the vorticity vector field, take the z-component [:, 2], and apply the mask
        data_matrix[:, i] = pt.masked_select(loader.load_snapshot("vorticity", time)[:, 2], mask)
    # subtract the temporal mean
    data_matrix -= pt.mean(data_matrix, dim=1).unsqueeze(-1)
    print(data_matrix.shape)
    pt.save(window_times,f"{data_save}window_times.pt")
    return data_matrix

#pat = "/home/jan/POD-ROM-fluid-flows/run/Re56"                 # erstmal weggelassen, da moden ein nicht periodisches Verhalten auffweisen -> noch keine ausgebildete Wirbelstraße
#Re56 = modeMaker(pat)
pat = "/home/jan/POD-ROM-fluid-flows/run/Re142"
Re142 = modeMaker(pat)
pat = "/home/jan/POD-ROM-fluid-flows/run/Re198"
Re198 = modeMaker(pat)
pat = "/home/jan/POD-ROM-fluid-flows/run/Re302"
Re302 = modeMaker(pat)
pat = "/home/jan/POD-ROM-fluid-flows/run/Re392"
Re392 = modeMaker(pat)

data_matrix = pt.cat((Re142,Re198,Re302,Re392), 0)     
svd = SVD(data_matrix)
modeCoeff = pt.zeros(1)
modeCoeff = svd.V#*svd.s    # Mode coefficients
#modeCoeff1 = modeCoeff[:len(Re56)]
#ReTensor = pt.zeros([len(modeCoeff),1])
#ReTensor += 56
#modeCoeff1 = pt.cat((ReTensor,modeCoeff),1)
modeCoeff2 = modeCoeff[:len(Re142)]                         #modeCoeff2 = modeCoeff[len(Re56):len(Re142)]
ReTensor = pt.zeros([len(modeCoeff),1])
ReTensor += 142
modeCoeff2 = pt.cat((ReTensor,modeCoeff),1)
modeCoeff3 = modeCoeff[len(Re142):len(Re198)]
ReTensor = pt.zeros([len(modeCoeff),1])
ReTensor += 198
modeCoeff3 = pt.cat((ReTensor,modeCoeff),1)
modeCoeff4 = modeCoeff[len(Re198):len(Re302)]
ReTensor = pt.zeros([len(modeCoeff),1])
ReTensor += 302
modeCoeff4 = pt.cat((ReTensor,modeCoeff),1)
modeCoeff5 = modeCoeff[len(Re302):len(Re392)]
ReTensor = pt.zeros([len(modeCoeff),1])
ReTensor += 392
modeCoeff5 = pt.cat((ReTensor,modeCoeff),1)

## modeCoeff wieder auseinanderschneiden um Re Zahl in Vektor einzufügen/oder Re im zusammengebautem Zustand einsetzem
modeCoeff = pt.cat(( modeCoeff2, modeCoeff3, modeCoeff4, modeCoeff5),1)
print(modeCoeff.shape)

pt.save(svd.s,f"{data_save}svds.pt")
pt.save(modeCoeff,f"{data_save}modeCoeff.pt")

# SVD fürt alle re gleichzeitg ausführen und dann trennen und Re in vektr einfügen
#-> wegn rekonstirklion, besonders für interpolation
print("alls klar")