#!/usr/bin/env python3
import torch as pt

def lhs_sampling(x_min, x_max, n_samples):
    assert len(x_min) == len(x_max)
    n_parameters = len(x_min)
    samples = pt.zeros((n_parameters, n_samples))
    for i, (lower, upper) in enumerate(zip(x_min, x_max)):
        bounds = pt.linspace(lower, upper, n_samples+1)
        rand = bounds[:-1] + pt.rand(n_samples) * (bounds[1:]-bounds[:-1])
        samples[i, :] = rand[pt.randperm(n_samples)]
    return samples


def change_line(path, line, lineNumber):
    setBoundary_file = open(path, "r")      
    lines = setBoundary_file.readlines()       
    lines[lineNumber] = line
    setBoundary_file = open(path, "w")
    setBoundary_file.writelines(lines)
    setBoundary_file.close()

def myround(x, base=5):
    return base * round(x/base)