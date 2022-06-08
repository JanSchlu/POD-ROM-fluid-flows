# POD-ROM-fluid-flows
## Overview
Prediction of time dependent fluid flows is a complicated simulation task. Reduced order models (ROM) are a way to overcome the lack of computing power. In this study, the POD-modes 

## Dependencies
python3, flowtorch,  
## Run
Load the flowtorch dataset 'of_cylinder2D_binary' in the parent directory. 
Create the dictionaries: data, plot. They will be used in the code.
Run SVD.py to extract the POD-modes out of the dataset.
Run defineAndTrain.py to defeine the neural network and train the model on the given dataset.
Run predictAndPlot.py to test the prediction and have a look on nice graphs.
