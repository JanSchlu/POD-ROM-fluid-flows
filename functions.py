#!/usr/bin/env python3.8

from re import S
import torch as pt
from os.path import isdir
import pickle
import os

from params import ReInput, SVD_modes, p_steps, data_save, n_inputs

class testNN(pt.nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
    def forward(self, x):
        return self.labels(x)


class FirstNN(pt.nn.Module):
    """Simple fully-connected neural network.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.n_inputs = kwargs.get("n_inputs", 1)
        self.n_outputs = kwargs.get("n_outputs", 1)
        self.n_layers = kwargs.get("n_layers", 1)
        self.n_neurons = kwargs.get("n_neurons", 10)
        self.activation = kwargs.get("activation", pt.sigmoid)
        self.layers = pt.nn.ModuleList()
        # input layer to first hidden layer
        self.layers.append(pt.nn.Linear(self.n_inputs, self.n_neurons, bias=True))#False))
        # add more hidden layers if specified
        if self.n_layers > 1:
            for hidden in range(self.n_layers-1):
                self.layers.append(pt.nn.Linear(self.n_neurons, self.n_neurons))
        # last hidden layer to output layer
        self.layers.append(pt.nn.Linear(self.n_neurons, self.n_outputs))
        #self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, pt.nn.Linear):
            module.weight.data.normal_(mean=1.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        for i_layer in range(len(self.layers)-1):
            x = self.activation(self.layers[i_layer](x))
        return self.layers[-1](x)

    @property
    def model_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def optimize_model(model: pt.nn.Module, input_train: pt.Tensor, output_train: pt.Tensor, epochs: int=1000, lr: float=0.001, save_best: str=""): 
 
    criterion = pt.nn.MSELoss()
    optimizer = pt.optim.Adam(params=model.parameters(), lr=lr)
    best_train_loss = 1.0e5
    train_loss = []
    for e in range(1, epochs+1):
        optimizer.zero_grad()
        prediction =  model(input_train)#.squeeze()                  # x_train ein Zeitschrit
        loss = criterion(prediction, output_train)	# y_train ist der nächste Zeitschritt
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        print("\r", "Training loss epoch {:5d}: {:10.5e}".format(
            e, train_loss[-1]), end="")
        if isdir(save_best):    
            if train_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_best}best_model_train.pt")
                best_train_loss = train_loss[-1]
    return train_loss

def rearrange_data(data: pt.tensor, steps: int=0):
    """Rearrange data tensor to input tensor and output tensor.

    The number of predictiable timesteps depents on the number of timesteps used as input (steps != 0).
    The number of predictiable timesteps is always one timestep less than the input timesteps.
     
    """
    if data.dim() == 1:
        input = pt.zeros(data.size(dim=0)-steps-1)
        output = pt.zeros(len(data)-steps-1)
    else:
        input = pt.zeros(data.size(dim=0)-steps-1, data.size(dim=1)+steps*data.size(dim=1))
        output = pt.zeros(len(data)-steps-1, data.size(dim=1))
    if steps == 0:
        input = data                                         
        output = data
        return input, output                              

    for timestep in range (len(data)-steps-1):                             #loop over all timesteps
        for next_timestep in range (0,steps):                               #add next timesteps to first step
            if next_timestep ==0:                                           #add first timestep to tensor
                help_tensor = data[timestep]
            y = pt.cat((help_tensor,data[timestep + next_timestep+1]))    #add next timestep to tensor
            help_tensor = y
        input[timestep]= y
        
    for i in range (len(output)):
        output[i] = data[i + steps+1]
    return input, output


def recalculate_output(output_data, string, deltaT):
    output = pt.zeros(len(output_data)-1, len(output_data[0]))                           
    if string == "sequential":
        for i in range (1,len(output)):
            output[i] = output_data[i+1]
        return output 
    if string == "residual":
        for i in range (1,len(output)):
            output[i] = output_data[i+1]-output_data[i]
        return output
    if string == "backward":
        for i in range (1,len(output)):
            output[i] = (3*output_data[i+1] - 4*output_data[i] + output_data[i-1])/(2*deltaT)
        return output

def predictor_singlestep(model, data_normIn):
    predict = pt.ones([len(data_normIn)-p_steps-2,SVD_modes])                                       
    for i in range (0, len(predict)):
        predict[i] = model(data_normIn[i]).squeeze()               
    predict = predict.detach().numpy()
    return predict


def predictor_sequential(model, data_norm_x,scalerdic):
    InScaler = MinMaxScaler()
    InScaler.refit(scalerdic["MinInS"], scalerdic["MaxInS"], ReInput)
    OutScaler = MinMaxScaler()
    OutScaler.refit(scalerdic["MinOutS"], scalerdic["MaxOutS"], ReInput)
    predicted_x = pt.ones([data_norm_x.shape[0],data_norm_x.shape[1]])    
    predicted_x[0] = data_norm_x[0]
    #predicted_x[0][0] = 1
    for i in range (0, len(predicted_x)-1):
        predictor = InScaler.scale(OutScaler.rescale(model(predicted_x[i]).squeeze()))                          
        if ReInput == True:
            test = pt.tensor([1])
            predictor = pt.cat((test,predictor),0)            
            predicted_x[i+1] = pt.cat((predicted_x[i,n_inputs:],predictor))
        predicted_x[i+1] = pt.cat((predicted_x[i,SVD_modes*p_steps:],predictor))            
    predicted_x = predicted_x.detach().numpy()
    return predicted_x


def predictor_residual(model, data_norm_x,scalerdic):
    InScaler = MinMaxScaler()
    InScaler.refit(scalerdic["MinInR"], scalerdic["MaxInR"], ReInput)
    OutScaler = MinMaxScaler()
    OutScaler.refit(scalerdic["MinOutR"], scalerdic["MaxOutR"], ReInput)          
    predicted_x = pt.ones([data_norm_x.shape[0],data_norm_x.shape[1]])   
    predicted_x[0] = data_norm_x[0]
    #predicted_x[0][0] = 1
    for i in range (0, len(predicted_x)-1):
        predictor = OutScaler.rescale(model(predicted_x[i]).squeeze())                                    
        if ReInput == True:
            predictor = InScaler.rescale(predicted_x[i,1+n_inputs*p_steps:]) + predictor  
            test = pt.tensor([1])
            predictor = pt.cat((test,predictor),0)
            predictor = InScaler.scale(predictor)                                      
            predictor[0] = 1
            predicted_x[i+1] = pt.cat((predicted_x[i,n_inputs:],predictor))             
        else:       
            predictor = InScaler.rescale(predicted_x[i,SVD_modes*p_steps:]) + predictor  
            predictor = InScaler.scale(predictor)                                       
            predicted_x[i+1] = pt.cat((predicted_x[i,SVD_modes:],predictor))          
    predicted_x = predicted_x.detach().numpy()
    return predicted_x

def predictor_backward(model, data_norm_x, scalerdic):
    InScaler = MinMaxScaler()
    InScaler.refit(scalerdic["MinInBW"], scalerdic["MaxInBW"], ReInput)
    OutScaler = MinMaxScaler()
    OutScaler.refit(scalerdic["MinOutBW"], scalerdic["MaxOutBW"], ReInput)                  
    predicted_x = pt.ones([data_norm_x.shape[0],data_norm_x.shape[1]])
    predicted_x[0] =data_norm_x[0]
    #predicted_x[0][0] = 1    
    predicted_x[1] =data_norm_x[1]  
    #predicted_x[1][0] = 1
    y_BW = pt.ones([data_norm_x.shape[0],SVD_modes])
    data = InScaler.rescale(data_norm_x)
    if ReInput == True:
        y_BW[0]=data[0,1+SVD_modes*p_steps:]  
        y_BW[1]=data[1,1+SVD_modes*p_steps:]         
    else:
        y_BW[0]=data[0,SVD_modes*p_steps:]
        y_BW[1]=data[1,SVD_modes*p_steps:]   
  
    for i in range(1,len(predicted_x)-1):
        y = OutScaler.rescale(model(predicted_x[i]).squeeze())
        y_BW[i+1]= 4/3*y_BW[i] -1/3*y_BW[i-1] + 2/3*y*5e-3          
        if ReInput == True:
            predictor = pt.cat((predicted_x[i,1+SVD_modes:],InScaler.scale(y_BW[i+1])))
            test = pt.tensor([1])
            predicted_x[i+1] = pt.cat((test,predictor),0)
        else:        
            predicted_x[i+1] = pt.cat((predicted_x[i,SVD_modes:],InScaler.scale(y_BW[i+1])))
    predicted_x = predicted_x.detach().numpy()
    return predicted_x    


def dataManipulator(modeData,timestep, string):
    
    ###### Längen der Tensoren für Training und Validierung aus Länge der Daten bestimmen und speichern
    maxLen = (len(modeData))
    lenTrain = int(maxLen * 2 / 3)    # training data sind die ersten 2/3 Zeitschritte mit SVD_modes Koeffizienten [batch,Anzahl Coeff]
    lenTest = maxLen - lenTrain
    pt.save(lenTrain, f"{data_save}lenTrain.pt")
    pt.save(lenTest, f"{data_save}lenTest.pt")
    
    ###### Daten aufteilen(gewünsche Anzahl an Moden), umsortieren und Vergeleichswerte(y) berechnen in abh vom Differenzenschemata
    inputs = SVD_modes
    if ReInput == True:
        inputs = SVD_modes +1 
    TrainData = modeData[:lenTrain,:inputs]
    TestData = modeData[lenTrain:,:inputs]

    train_data, y_train = rearrange_data(TrainData, p_steps)
    y_train = recalculate_output(y_train,string,timestep)
    test_data, y_test = rearrange_data(TestData,p_steps)
    y_test = recalculate_output(y_test,string,timestep)
    train_data=train_data[1:-1]
    test_data=test_data[1:-1]
    y_train=y_train[1:]
    y_test=y_test[1:]
    if ReInput == True:
        y_train=y_train[:,1:]
        y_test=y_test[:,1:]
    return train_data, y_train, test_data, y_test

class MinMaxScaler(object):
    """Class to scale/re-scale data to the range [-1, 1] and back.
    """
    def __init__(self):
        self.min = None
        self.max = None
        self.trained = False

    def fit(self, data, ReInput):
        self.min = data.min(dim=0).values
        self.max = data.max(dim=0).values
        self.ReIn = ReInput
        self.trained = True

    def refit(self, min, max, ReInput):
        self.min = min
        self.max = max
        self.ReIn = ReInput
        self.trained = True

    def scale(self, data):
        assert self.trained
        #assert len(data.shape) == 2
        data_norm = data
        length = data_norm.shape[0]
        if len(data_norm.shape) == 2:
            length = data_norm.shape[1]
        min = self.min
        max = self.max        
        if self.ReIn == True:
            if  length == len(self.min)-1:
                min = self.min[1:SVD_modes+1]
                max = self.max[1:SVD_modes+1] 
            if length == len(self.min):
                min = self.min
                max = self.max     
        if len(data.shape) == 2:
            for i in range (1,int((data.shape[1]-SVD_modes)/SVD_modes)):
                min = pt.cat((min, self.min[:SVD_modes]))                   ## falsch
                max = pt.cat((max, self.max[:SVD_modes]))                   ##
        if length == SVD_modes:
            min = self.min[:SVD_modes]
            max = self.max[:SVD_modes]                
        data_norm = (data - min) / (max - min)
        return data_norm#2.0*data_norm - 1.0

    def rescale(self, data_norm):
        assert self.trained
#        assert len(data_norm.shape) == 2
        data = data_norm
        length = data_norm.shape[0]
        if len(data_norm.shape) == 2:
            length = data_norm.shape[1]
        min = self.min
        max = self.max    
        if self.ReIn == True:
            if  length == len(self.min)-1:
                min = self.min[1:SVD_modes+1]
                max = self.max[1:SVD_modes+1] 
            if length == len(self.min):
                min = self.min
                max = self.max   
        if len(data.shape) == 2:
            for i in range (1,int((data.shape[1]-SVD_modes)/SVD_modes)):
                min = pt.cat((min, self.min[:SVD_modes]))                   ## falsch
                max = pt.cat((max, self.max[:SVD_modes]))                   ##
        if length == SVD_modes:
            min = self.min[:SVD_modes]
            max = self.max[:SVD_modes]         

        data = data_norm* (max - min) + min
        return data#2.0*data_norm - 1.0
    
    def save(self):
        return self.min, self.max

def dataloader(path):
    if os.path.isfile(path) == True:
        with open(path, "rb") as input:
            data = pickle.load(input)
    else: data = {}
    return data