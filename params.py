import torch as pt
data_save = "/home/jan/POD-ROM-fluid-flows/run/data/"

p_steps =   2        #Anzahl der mitberücksichtigen vorrangegagenen zeitschritte
SVD_modes = 10 # first 19 singular values yield 99.12%
n_inputs = (p_steps + 1) *  SVD_modes

model_params = {
"n_inputs": n_inputs,
"n_outputs": SVD_modes,
"n_layers": 2,
"n_neurons": 80,
"activation": pt.nn.ReLU()	#fast and accurate
}