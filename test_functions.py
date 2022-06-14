import torch as pt
from functions import rearrange_data, split_data, subtract_data
def test_rearrange_data():
    data =pt.tensor([[1, 2],[3, 4],[5, 6],[7, 8]])

    input0=pt.tensor([[1, 2],[3, 4],[5, 6]])
    output0=pt.tensor([[3, 4],[5, 6],[7, 8]])
    input, output = rearrange_data(data,0)
    assert pt.max(input0-input) == 0
    assert pt.max(output0-output) == 0

    input1=pt.tensor([[1, 2, 3, 4],[3, 4, 5, 6]])
    output1=pt.tensor([[5, 6],[7, 8]])
    input, output = rearrange_data(data,1)
    assert pt.max(input1-input) == 0
    assert pt.max(output1-output) == 0 

    input2=pt.tensor([[1, 2, 3, 4, 5, 6]])
    output2=pt.tensor([[7, 8]])
    input, output = rearrange_data(data,2)
    assert pt.max(input2-input) == 0
    assert pt.max(output2-output) == 0

def test_split_data():
    data = pt.tensor([[1, 2, 3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21]])
    data_check = pt.tensor([[7,8],[10,11]])
    data_vgl = split_data(data, timesteps_out=2, modes_out=2, timesteps_skip=2)

    assert pt.max(data_check - data_vgl) == 0

def test_subtract_data():
    data = pt.tensor([[1, 2, 3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21]])
    data_check = pt.tensor([[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]])
    data_vgl = subtract_data(data)

    assert pt.max(data_check - data_vgl) == 0