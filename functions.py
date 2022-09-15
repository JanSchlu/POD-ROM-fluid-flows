
import torch as pt
from os.path import isdir
import pickle
import os

from params import SVD_modes, p_steps, data_save

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
      #  print(input_train.shape)
        prediction =  model(input_train)#.squeeze()                  # x_train ein Zeitschrit
        loss = criterion(prediction, output_train)	# y_train ist der nächste Zeitschritt
      #  print(prediction.shape, output_train.shape, input_train.shape)
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
        for i in range (len(output)):
                input[i] = data[i]                                         
                output[i] = data[i + 1]
   
        return input, output                              

    for timestep in range (len(data)-steps-1):                             #loop over all timesteps
        for next_timestep in range (0,steps):                               #add next timesteps to first step
            if next_timestep ==0:                                           #add first timestep to tensor
                help_tensor = data[timestep]
            y = pt.cat((help_tensor,data[timestep + next_timestep+1]))    #add next timestep to tensor
            help_tensor = y
        input[timestep]= y

    for i in range (len(output)):
        output[i] = data[i + steps +1]
          
    return input, output


def recalculate_output(input_data,output_data, string, deltaT):
    lastStepBegin= SVD_modes*p_steps                            # auf 3 stellen für pytest
    output = pt.zeros(len(output_data)-1, len(output_data[0]))
    if string == "sequential":
        return output_data[:-1]
    if string == "residual":
        output[0] = output_data[0]-input_data[0,lastStepBegin:]
        for i in range (1,len(output)):
            output[i] = output_data[i]-output_data[i-1]
        return output
    if string == "backward":
        output[0] = (3*output_data[1] - 4*output_data[0] + input_data[0,lastStepBegin:])/(2*deltaT)
        for i in range (1,len(output)):
            output[i] = (3*output_data[i+1] - 4*output_data[i] + output_data[i-1])/(2*deltaT)
        return output

def recalculate_output_back(input_data,output_data, string, deltaT):
    lastStepBegin= SVD_modes*p_steps                            # auf 3 stellen für pytest
    output = pt.zeros(len(output_data)-1, len(output_data[0]))
    if string == "sequential":
        return output_data[:-1]
    if string == "residual":
        output[0] = output_data[0]-input_data[0,lastStepBegin:]
        for i in range (1,len(output)):
            output[i] = output_data[i]-output_data[i-1]
        return output
    if string == "backward":
        output[0] = (3*output_data[1] - 4*output_data[0] + input_data[0,lastStepBegin:])/(2*deltaT)
        for i in range (1,len(output)):
            output[i] = (3*output_data[i+1] - 4*output_data[i] + output_data[i-1])/(2*deltaT)
        return output



        
def split_data(data_in: pt.tensor, timesteps_out: int, modes_out: int, timesteps_skip: int = 0):
    """Return timesteps for the number of modes given. 

    :timesteps_out: number of timesteps returned
    :timesteps_skip: number of skipt timesteps
    """
    data_out = pt.zeros([timesteps_out,modes_out])
    for i in range (timesteps_out):
        for n in range (0,modes_out):
            data_out[i , n] = data_in[i + timesteps_skip, n]
    return data_out



def predictor_sequential(model, data):
    predict = pt.ones([len(data)-p_steps-2,SVD_modes])                                           # pred len(test_data)-1-p_steps
    for i in range (0, len(predict)):
        predict[i] = model(data[i]).squeeze()               # model predict data[i+1]
    predict = predict.detach().numpy()
    return predict

def predictor_residual(model, data_normIN):
    predict = pt.zeros([len(data_normIN)-p_steps-2,SVD_modes])                                         # pred len(test_data)-1-p_steps
    for i in range (0, len(predict)):
        #predict[0] = data_normOUT[1]
        if i>0:                 
            predict[i] = model(data_normIN[i]).squeeze()
    predict = predict.detach().numpy()
    return predict

def predictor_backward(model, data_normIN, min_x, max_x, min_y, max_y):
    predict = pt.ones([len(data_normIN)-p_steps-2,SVD_modes])                                         # pred len(test_data)-1-p_steps
    print(len(predict),len(data_normIN))
    for i in range (0, len(predict)):
        #predict[0] = (data_normIN[1]* (max_x - min_x) + min_x)
        #predict[1] = (data_normIN[2]* (max_x - min_x) + min_x)
        #if i>0:
        predict[i] = (model(data_normIN[i]).squeeze())# 4/3*predict[i-1] - 1/3*predict[i-2] + 2/3*((max_y-min_y)*(model(data_normIN[i]).squeeze())+min_y)*5e-3 # 5e-3 is timestepssize
    predict = predict.detach().numpy()
    return predict


def predictor_sequential_period(model, data, min_x, max_x, min_y, max_y):
    predStore = pt.ones([len(data)-p_steps,SVD_modes+SVD_modes*p_steps])                 
    predicted = pt.ones([len(data)-p_steps-1,SVD_modes])
    predStore[0] = data[0]                  # muss in x normiert sein                                             #start is last timestep of trainData
    predicted[0] = ((data[0, SVD_modes*p_steps:]*(max_x - min_x) + min_x)-min_y)/(max_y-min_y)      # muss in y normiert sein
    for i in range (1, len(predStore)):
        prediction = ((model(predStore[i-1]).squeeze()*(max_y - min_y) + min_y)-min_x)/(max_x - min_x)      # muss in x normerit sein -> wird in predStore gesepichertz
        predStore[i] = pt.cat((predStore[i-1,SVD_modes:], prediction))
        predicted[i-1] = prediction 
    predicted = predicted.detach().numpy()
    return predicted

def predictor_residual_period(model, data, min_x, max_x, min_y, max_y):
    predStore = pt.ones([len(data)-p_steps,SVD_modes+SVD_modes*p_steps])                 
    predicted = pt.ones([len(data)-p_steps-1,SVD_modes])
    predStore[0] = data[0]                                                               #start is last timestep of trainData
    predicted[0] = ((data[0, SVD_modes*p_steps:]*(max_x - min_x) + min_x)-min_y)/(max_y-min_y)
    for i in range (1, 100):#len(predStore)):
        prediction = (((model(predStore[i-1]).squeeze())*(max_y - min_y) + min_y)-min_x)/(max_x - min_x)  #predStore[i-1,p_steps*20:] + 
        print("1",prediction)
        print("2",predStore[i-1,SVD_modes:])
        predStore[i] = pt.cat((predStore[i-1,SVD_modes:], prediction))
        print("3",predStore[i])
        predicted[i-1] = prediction 
    predicted = predicted.detach().numpy()
    return predicted
    
def predictor_backward_period(model, data, min, max):
    predStore = pt.ones([len(data)-p_steps,SVD_modes+SVD_modes*p_steps])                 
    predicted = pt.ones([len(data)-p_steps-1,SVD_modes])
    predStore[0] = data[0]                                                               #start is last timestep of trainData
    predicted[0] = data[0, SVD_modes*p_steps:]

    predict = pt.ones([len(data)-p_steps,SVD_modes])                                         # pred len(test_data)-1-p_steps
    for i in range (0, len(predict)):
        predict[0] = data[1, SVD_modes*p_steps:]
        predict[1] = data[2, SVD_modes*p_steps:]
        if i>1:
            predict[i] = 4/3*predict[i-1] - 1/3*predict[i-2] + 2/3*(model(data[i]).squeeze()*(max - min) + min)*5e-3 # 5e-3 is timestepssize
    predict = predict.detach().numpy()
    
    for i in range (1, len(predStore)):
        prediction = 4/3*predStore[i-1,p_steps*20:] - 1/3*predStore[i-1,p_steps*10:p_steps*10+10] + 2/3*(model(predStore[i-1]).squeeze()*(max - min) + min)*5e-3
        predStore[i] = pt.cat((predStore[i-1,SVD_modes:], prediction))
        predicted[i-1] = prediction 
    predicted = predicted.detach().numpy()
    return predicted


#def preditorOfPredicted(data, scheme):
    predStore = pt.ones([len(data)-p_steps,SVD_modes+SVD_modes*p_steps])                 
    predicted = pt.ones([len(data)-p_steps,SVD_modes])
    predStore[0] = data[0]                                                               #start is last timestep of trainData
    predicted[0] = data[0, SVD_modes*p_steps:]
    for i in range (1, len(predStore)):
        if scheme == "sequential":
            prediction = best_model(predStore[i-1]).squeeze()
        if scheme == "residual":
            prediction = predStore[i-1,p_steps*20:]+best_modelR(predStore[i-1]).squeeze()
        if scheme == "backward":
            prediction = 4/3*predStore[i-1,p_steps*20:] - 1/3*predStore[i-1,p_steps*10:p_steps*10+10] + 2/3*best_modelBW(predStore[i-1]).squeeze()*5e-3
        predStore[i] = pt.cat((predStore[i-1,SVD_modes:], prediction))
        predicted[i-1] = prediction 
    predicted = predicted.detach().numpy()
    return predicted




def dataManipulator(modeData, modenumbers, steps, string):

    ###### normierung
    #minData = modeData.min(dim=0).values
    #maxData = modeData.max(dim=0).values
    #modeData = (modeData - minData)/(maxData-minData)
    #pt.save(minData, f"{data_save}minCoeff.pt")
    #pt.save(maxData, f"{data_save}maxCoeff.pt")
    
    ###### Längen der Tensoren für Training und Validierung aus Länge der Daten bestimmen und speichern
    maxLen = (len(modeData))
    lenTrain = int(maxLen * 2 / 3)    # training data sind die ersten 2/3 Zeitschritte mit SVD_modes Koeffizienten [batch,Anzahl Coeff]
    lenTest = maxLen - lenTrain
    pt.save(lenTrain, f"{data_save}lenTrain.pt")
    pt.save(lenTest, f"{data_save}lenTest.pt")
    print(maxLen, lenTrain, lenTest)
    
    ###### Daten aufteilen(gewünsche Anzahl an Moden), umsortieren und Vergeleichswerte(y) berechnen in abh vom Differenzenschemata
    train_data, y_train = rearrange_data(split_data(modeData, lenTrain, modenumbers,0), steps)
    y_train = recalculate_output(train_data,y_train,string,5e-3)
    train_data=train_data[:-1]
    test_data, y_test = rearrange_data(split_data(modeData, lenTest, modenumbers, lenTrain), steps)
    y_test = recalculate_output(test_data,y_test,string,5e-3)
        
    return train_data, y_train, test_data, y_test


def dataManipulator_yTest(modeData, modenumbers, steps):

    ###### normierung
    #minData = modeData.min(dim=0).values
    #maxData = modeData.max(dim=0).values
    #modeData = (modeData - minData)/(maxData-minData)
    #pt.save(minData, f"{data_save}minCoeff.pt")
    #pt.save(maxData, f"{data_save}maxCoeff.pt")
    
    ###### Längen der Tensoren für Training und Validierung aus Länge der Daten bestimmen und speichern
    lenTrain = pt.load (f"{data_save}lenTrain.pt")
    lenTest = pt.load (f"{data_save}lenTest.pt")
    
    ###### Daten aufteilen(gewünsche Anzahl an Moden), umsortieren und Vergeleichswerte(y) berechnen in abh vom Differenzenschemata
    test_data, y_test = rearrange_data(split_data(modeData, lenTest, modenumbers, lenTrain), steps)
    #y_test = recalculate_output(test_data,y_test,string,5e-3)
        
    return test_data, y_test





    # -> TEST:  mit mehr SVD modes
    # -> TEST:  mit Fehlerpfortpflazung
    # -> TEST:  mit richtigen Daten

class MinMaxScaler(object):
    """Class to scale/re-scale data to the range [-1, 1] and back.
    """
    def __init__(self):
        self.min = None
        self.max = None
        self.trained = False

    def fit(self, data):
        self.min = data.min(dim=0).values
        self.max = data.max(dim=0).values
        self.trained = True

    def scale(self, data):
        assert self.trained
        assert len(data.shape) == 2
        data_norm = (data - self.min) / (self.max - self.min)
        return data_norm#2.0*data_norm - 1.0

    def rescale(self, data_norm):
        assert self.trained
        assert len(data_norm.shape) == 2
        data = data_norm#(data_norm + 1.0) * 0.5
        return data * (self.max - self.min) + self.min
    
    def save(self):
        return self.min, self.max

    def restore(self, min, max):
        self.min = min
        self.max = max
        self.trained = True


def dataloader(path):
    if os.path.isfile(path) == True:
        with open(path, "rb") as input:
            data = pickle.load(input)
    else: data = {}
    return data