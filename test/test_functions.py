from numpy import rec
import torch as pt
from functions import rearrange_data, split_data, recalculate_output
from params import SVD_modes
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

def test_recalculate_output():
    data = pt.tensor([[1, 2, 3],[7,8,9],[4,5,6],[10,11,12],[16,17,18],[13,14,15],[19,20,21]])
    input_data = pt.tensor([[1,2,3,7,8,9],[4,5,6,10,11,12],[16,17,18,13,14,15]])

    data_check = pt.tensor([[1, 2, 3],[7,8,9],[4,5,6],[10,11,12],[16,17,18],[13,14,15]])
    data_vgl = recalculate_output(input_data,data,"sequential")
    assert pt.max((data_check - data_vgl)**2) == 0

    data_check = pt.tensor([[-6,-6,-6],[6,6,6],[-3,-3,-3],[6,6,6],[6,6,6],[-3,-3,-3]])
    data_vgl = recalculate_output(input_data,data,"residual")
    assert pt.max((data_check - data_vgl)**2) == 0

    data_check = pt.tensor([[2400,2400,2400],[-1500,-1500,-1500],[2100,2100,2100],[1200,1200,1200],[-1500,-1500,-1500],[2100,2100,2100]])
    data_vgl = recalculate_output(input_data,data,"backward")

    assert pt.max((data_check - data_vgl)**2) == 0


def predictor_sequential(data):
    predict = pt.ones([len(data)-1,1])                                           # pred len(test_data)-1-p_steps
    for i in range (0, len(predict)):
        predict[i] = data[i+1]
    predict = predict.detach().numpy()
    return predict

def predictor_residual(data):
    predict = pt.ones([len(data)-1,1])                                           # pred len(test_data)-1-p_steps
    for i in range (0, len(predict)):
        predict[0] = data[1]
        if i>0:
            predict[i] = data[i] + 0.01 # modelR(data[i])
    predict = predict.detach().numpy()
    return predict

def predictor_backward(data):
    predict = pt.ones([len(data)-1,1])                                           # pred len(test_data)-1-p_steps
    for i in range (0, len(predict)):
        predict[0] = data[1]
        predict[1] = data[2]
        if i>1:
            predict[i] = 4/3*data[i] - 1/3*data[i-1] + 2/3*0.01*1
    predict = predict.detach().numpy()
    return predict