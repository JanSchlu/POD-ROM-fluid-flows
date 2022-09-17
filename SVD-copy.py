#!/usr/bin/env python3.8
"""
code for SVD, uses data from flowtorch -> of_cylinder2d_binary
SVD for multiple datasets
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
    pt.save(window_times,f"{data_save}window_times.pt")
    return data_matrix

pat = "/home/jan/POD-ROM-fluid-flows/run/Re56"
Re56 = modeMaker(pat)
pat = "/home/jan/POD-ROM-fluid-flows/run/Re142"
Re142 = modeMaker(pat)
pat = "/home/jan/POD-ROM-fluid-flows/run/Re302"
Re198 = modeMaker(pat)
pat = "/home/jan/POD-ROM-fluid-flows/run/Re302"
Re302 = modeMaker(pat)
pat = "/home/jan/POD-ROM-fluid-flows/run/Re302"
Re392 = modeMaker(pat)

data_matrix = pt.cat((Re56,Re142,Re198,Re302,Re392), 0)     
svd = SVD(data_matrix)
modeCoeff = pt.zeros(1)
modeCoeff = svd.V#*svd.s    # Mode coefficients
## modeCoeff wieder auseinanderschneiden um Re Zahl in Vektor einzufügen/oder Re im zusammengebautem Zustand einsetzem

ReTensor = pt.zeros([len(modeCoeff),1])
ReTensor += ReNumber
modeCoeff = pt.cat((ReTensor,modeCoeff),1)

pt.save(svd.s,f"{data_save}svds.pt")
pt.save(modeCoeff,f"{data_save}modeCoeff.pt")

# SVD fürt alle re gleichzeitg ausführen und dann trennen und Re in vektr einfügen
#-> wegn rekonstirklion, besonders für interpolation

