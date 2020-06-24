#  SpaceNet-3 Data preparation
To prepare the SpaceNet-3 Road Detection Dataset you have to create a Python virtual environment and run the spacenet_prep.py script. Directions are given bellow.


## Installation


1. Create a virtual environment 
```
$ python3 -m venv /path/to/new/virtual/env
```
2. Tell pip to install all of the dependencies using the `requirements.txt` file:
```
$ pip install -r requirements.txt
```

## Running the scripts

Before running the scripts to prepare the dataset, please open the `config.json` file and modify the data directories according to your needs. After making sure that you provided all the necessary directories, you can run the script to prepare the dataset. 

```
python spacenet_prep.py
```