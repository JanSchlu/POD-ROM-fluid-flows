# POD-ROM-fluid-flows
## Overview
Prediction of time dependent fluid flows is a complicated simulation task. Reduced order models (ROM) are a way to overcome the lack of computing power. In this study, the POD-modes 

## Dependencies
python3, flowtorch, singularity
## Run
Create a run/ dictionary.
Create in run/ the dictionaries: data/, plot/. They will be used in the code.
Run training/SVD.py to extract the POD-modes out of the dataset.
Run a train script (train_...) to define the neural network and train the model on the given dataset.
Run /prediction/prediction.py to test the prediction and have a look on nice graphs.

## Parameter Study
Das Modell reduzierter Ordnung soll mit der Reynoldszahl parametrisiert werden. Das Skript runParameterStudy.py ermittellt im gewünschtem Re Bereich durch hypercube sampling die zu berehnenden Re Zahlen und erstellt für jede einen Ornder. In den jeweiligen Ordner wird der base_case, hier: cylinder2D_base, kopiert und die Geschwindigkeit der jeweiligen Re Zahl angepasst. Im Anschluss werden alle Re Fälle berechnet und die Ergebnisse in den Ordnern gespeichert.