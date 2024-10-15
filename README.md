# MlFlow tutorial



## Getting started

Make sure you have:

- [ ] Cloned this repository 
- [ ] A virtual environment already set-up with python >= 3.10
- [ ] Installed the requirements

An example using venv:
```
git clone https://github.com/danielm322/MLFlow_tutorial.git
cd mlflow_tutorial
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Or using miniforge (conda open source version):
```
git clone https://github.com/danielm322/MLFlow_tutorial.git
cd mlflow_tutorial
conda create -n mlflow_tuto_env python=3.10 
conda activate mlflow_tuto_env
pip install -r requirements.txt
```
***

## Connect to the MLFlow server for the tutorial
Open this address in your web browser: `http://10.8.33.50:8085/` (Only accessible through the intranet)


## Complete missing lines in the python script and run script
- [ ] Complete all the TODOs in the `mnist_mlflow_tutorial.py`. 
- [ ] Make sure that everyone in the session have given the same names to parameters and metrics before running the script!
- [ ] Make sure everyone sets a different value for the learning rate while keeping a fixed dropout rate, before running the script
- [ ] Make sure everyone sets a different value for the dropout rate while keeping a fixed learning rate, before running the script again

## Compare experiment runs in the UI
Go back to the User Interface to compare the experiments ran by the team!

## Running and logging to a local MLFlow server
Notice that if you want to log locally (solo work), you just need to remove or comment the line with `mlflow.set_tracking_uri("http://10.8.33.50:8085")` in the script.

Then to launch a local mlflow server, run `mlflow ui` in the base folder of the code.
