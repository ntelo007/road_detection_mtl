#  Road Detection from Remote Sensing Imagery

This is the official repository of the graduation project of Pantelis Kaniouras for the MSc Geomatics of Delft University of Technology, Netherlands with title *Road Detection from Remote Sensing Imagery*. The link to download the thesis will be available soon.


## Installation

To get a copy of the project up and running, clone or download the content of the repository and create an environment from the given environment.yml file. Use the terminal or an Anaconda Prompt for the following steps:


1. The first line of the yml file sets the new environment's name. If you want a specific name, go inside the `environment.yml` file and change it.

2. Create the environment from the `environment.yml` file:

```
conda env create -f environment.yml
```

3. Activate the new environment: 
```
conda activate segmodels
```

4. Verify that the new environment was installed correctly:
```
conda env list
```
You can also use `conda info --envs`.


## Running the scripts

The given python scripts contain functions both for training and testing the created models. Example execution: 

```
python unet_resnet.py --task two --loss cce_jaccard --encoder_weights yes 
```
or
```
python unet_mtl.py --task1 two --task2 intersection --task3 orientation --loss1 cce_jaccard --loss2 cce_jaccard --loss3 cce_jaccard
```



## Acknowledgments
Parts of the code were heavily derived from https://github.com/qubvel/segmentation_models and https://github.com/anilbatra2185/road_connectivity.

I would also like to thank Liangliang Nan and Frido Kuijper for their contribution in this project.
