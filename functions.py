from dataclasses import dataclass
import torch as pt
from os.path import isdir


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
        self.layers.append(pt.nn.Linear(self.n_inputs, self.n_neurons, bias=False))
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
    best_val_loss, best_train_loss = 1.0e5, 1.0e5
    train_loss, val_loss = [], []
    for e in range(1, epochs+1):
        optimizer.zero_grad()
        prediction =  model(input_train).squeeze()                  # x_train ein Zeitschrit
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
    input = pt.zeros(len(data)-steps-1, len(data[0])+steps*len(data[0]))       
    output = pt.zeros(len(data)-steps-1, len(data[0]))
    outputR = pt.zeros(len(data)-steps-1, len(data[0]))
    outputBW = pt.zeros(len(data)-steps-1, len(data[0]))
    if steps == 0:
        for i in range (len(output)):
            for n in range (0, len(data[0])):
                input[i, n] = data[i, n]                                         
                output[i, n] = data[i + 1, n]
                if i != 0 or i != (len(output)):
                    outputR[i,n] = data[i+1,n]-data[i, n]
                    outputBW[i,n] = (3*data[i+1,n]-4*data[i,n]+2*data[i-1,n])/2*5e-3
                else:
                    outputR[i,n] = outputBW[i,n] = output[i, n]
        return input, output, outputR, outputBW                                      # outputR and outputBW are missing, p_steps = 0 won't work 

    for timestep in range (len(data)-steps-1):                             #loop over all timesteps
        for next_timestep in range (0,steps):                               #add next timesteps to first step
            if next_timestep ==0:                                           #add first timestep to tensor
                help_tensor = data[timestep]
            y = pt.cat((help_tensor,data[timestep + next_timestep+1]))    #add next timestep to tensor
            help_tensor = y
        input[timestep]= y

    for i in range (len(output)):
        for n in range (0, len(data[0])):
            output[i, n] = data[i + steps +1, n]
            if i != 0 or i != (len(output)):
                outputR[i,n] = data[i+1,n]-data[i, n]
                outputBW[i,n] = (3*data[i+1,n]-4*data[i,n]+2*data[i-1,n])/2*5e-3
            else:
                outputR[i,n] = outputBW[i,n] = output[i, n]
    return input, output, outputR, outputBW

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
