import torch as pt
from functions import rearrange_data, recalculate_output, dataManipulator, MinMaxScaler
from params import ReInput
def test_rearrange_data():
    data =pt.tensor([[1, 2],[3, 4],[5, 6],[7, 8]])

    input, output = rearrange_data(data,0)
    assert pt.max(input-output) == 0

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

def test_recalculate_output():
    data = pt.tensor([[1, 2, 3],[7,8,9],[4,5,6],[10,11,12],[16,17,18],[13,14,15],[19,20,21]])
    input_data = pt.tensor([[1,2,3,7,8,9],[4,5,6,10,11,12],[16,17,18,13,14,15]])

    data_check = pt.tensor([[0,0,0],[4,5,6],[10,11,12],[16,17,18],[13,14,15],[19,20,21]])
    data_vgl = recalculate_output(data,"sequential",1)
    assert pt.max((data_check - data_vgl)**2) == 0

    data_check = pt.tensor([[0,0,0],[-3,-3,-3],[6,6,6],[6,6,6],[-3,-3,-3],[6,6,6]])
    data_vgl = recalculate_output(data,"residual",1)
    assert pt.max((data_check - data_vgl)**2) == 0

    data_check = pt.tensor([[0,0,0],[-7.5,-7.5,-7.5],[10.5,10.5,10.5],[6,6,6],[-7.5,-7.5,-7.5],[10.5,10.5,10.5]])
    data_vgl = recalculate_output(data,"backward",1)
    assert pt.max((data_check - data_vgl)**2) == 0

def test_dataManipulator():
    """
    ReInput = False in params

    """
    n_inputs = 2
    timestep = 1
    p_steps = 0
    data = pt.zeros(10,2)
    for i in range (1,10):
        data[i,0]=data[i-1,0]+0.01
        data[i,1]=1*i*i -13
        
    train_dataS, y_trainS, test_dataS, y_testS = dataManipulator(data, n_inputs, p_steps, timestep, "sequential") 
    train_dataR, y_trainR, test_dataR, y_testR = dataManipulator(data, n_inputs, p_steps, timestep, "residual") 
    train_dataBW, y_trainBW, test_dataBW, y_testBW = dataManipulator(data, n_inputs, p_steps, timestep, "backward") 

    train_data_vgl = data[1:len(y_trainS)+1]
    assert pt.max(train_dataS-train_data_vgl) == 0
    assert pt.max(train_dataR-train_data_vgl) == 0
    assert pt.max(train_dataBW-train_data_vgl) == 0

    test_data_vgl = data[len(y_trainS)+3:-1]       
    assert pt.max(test_dataS-test_data_vgl) == 0
    assert pt.max(test_dataR-test_data_vgl) == 0
    assert pt.max(test_dataBW-test_data_vgl) == 0

    y_train_1N_vgl = data[0:len(y_trainS)]
    y_train_N_vgl = data[1:len(y_trainS)+1]
    y_train_N1_vgl = data[2:len(y_trainS)+2]
    y_trainS_vgl = y_train_N1_vgl
    y_trainR_vgl = (y_train_N1_vgl - y_train_N_vgl)
    y_trainBW_vgl = (3*y_train_N1_vgl-4*y_train_N_vgl+y_train_1N_vgl)/(2*timestep)

    assert pt.max(y_trainS -y_trainS_vgl) == 0
    assert pt.max(y_trainR - y_trainR_vgl) == 0
    assert pt.max(y_trainBW - y_trainBW_vgl) == 0

    y_test_1N_vgl = data[len(y_trainS)+2:-2]
    y_test_N_vgl = data[len(y_trainS)+3:-1]
    
    y_test_N1_vgl = data[len(y_trainS)+4:]
    y_testS_vgl = y_test_N1_vgl
    y_testR_vgl = y_test_N1_vgl - y_test_N_vgl
    y_testBW_vgl = (3*y_test_N1_vgl-4*y_test_N_vgl+y_test_1N_vgl)/(2*timestep)

    assert pt.max(y_testS -y_testS_vgl) == 0
    assert pt.max(y_testR - y_testR_vgl) == 0
    assert pt.max(y_testBW - y_testBW_vgl) == 0
