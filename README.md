#  Road Detection from Remote Sensing Imagery

This is the official repository of the graduation project of Pantelis Kaniouras for the MSc Geomatics of Delft University of Technology, Netherlands with title *Road Detection from Remote Sensing Imagery*. By clicking on the following link, you can download the thesis and read more information about it: https://repository.tudelft.nl/islandora/object/uuid%3A21fc20a8-455d-4583-9698-4fea04516f03 

Abstract: *Road network maps facilitate a great number of applications in our everyday life. However, their automatic creation is a difficult task, and so far, published methodologies cannot provide reliable solutions. The common and most recent approach is to design a road detection algorithm from remote sensing imagery based on a Convolutional Neural Network, followed by a result refinement post-processing step. In this project I proposed a deep learning model that utilized the Multi-Task Learning technique to improve the performance of the road detection task by incorporating prior knowledge constraints. Multi-Task Learning is a mechanism whose objective is to improve a model's generalization performance by exploiting information retrieved from the training signals of related tasks as an inductive bias, and, as its name suggests, solve multiple tasks simultaneously. Carefully selecting which tasks will be jointly solved favors the preservation of specific properties of the target object, in this case, the road network. My proposed model is a Multi-Task Learning U-Net with a ResNet34 encoder, pre-trained on the ImageNet dataset, that solves for the tasks of Road Detection Learning, Road Orientation Learning, and Road Intersection Learning. Combining the capabilities of the U-Net model, the ResNet encoder and the constrained Multi-Task Learning mechanism, my model achieved better performance both in terms of image segmentation and topology preservation against the baseline single-task solving model. The project was based on the publicly available SpaceNet Roads Dataset.*

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

## Image of the Proposed Model's architecture for the case of solving for 2 tasks
![ScreenShot](/images/architecture_mtl.png)

## Image of the Training Procedure of the Proposed Model for the case of solving for 2 tasks
![ScreenShot](/images/mtl_2_example.png)

## Accuracy assessment results
Road Detection Task  | Road Intersection Task  | Road Orientation Task  | Road Gaussian Task  | Mean IoU  | Mean F1  | Mean clDice  | 
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
X  | -  | -  | -  | 0.684  | 0.732  | 0.671  | 
X  | X  | -  | -  | 0.677  | 0.725  | 0.661  | 
X  | -  | -  | X  | 0.679  | 0.726  | 0.661  |
X  | -  | X  | -  | 0.680  | 0.729  | 0.676  |
X  | X  | X  | -  | **0.691**  | **0.740**  | **0.698**  |
X  | X  | -  | X  | 0.681  | 0.727  | 0.657  |
X  | -  | X  | X  | 0.668  | 0.712  | 0.621  |
X  | X  | X  | X  | 0.673  | 0.716  | 0.621  |


## Prediction Results
Satellite Image  | Ground Truth Mask  | Single-task prediction  | MTL prediction
:---:|:---:|:---:|:---:
![Alt Text](/Results/10710_image.PNG) | ![Alt Text](/Results/10710_gt.PNG) | ![Alt Text](/Results/10710_two.PNG) | ![Alt Text](/Results/10710_mtl_result.PNG)
![Alt Text](/Results/1_image.PNG) | ![Alt Text](/Results/1_gt.PNG) | ![Alt Text](/Results/1_two.PNG) | ![Alt Text](/Results/1_mtl.PNG)
![Alt Text](/Results/3989_image.PNG) | ![Alt Text](/Results/3989_gt.PNG) | ![Alt Text](/Results/3989_two.PNG) | ![Alt Text](/Results/3989_mtl_result.PNG)
![Alt Text](/Results/3_image.PNG) | ![Alt Text](/Results/3_gt.PNG) | ![Alt Text](/Results/3_two.PNG) | ![Alt Text](/Results/3_mtl.PNG)
![Alt Text](/Results/4059_image.PNG) | ![Alt Text](/Results/4059_gt.PNG) | ![Alt Text](/Results/4059_two.PNG) | ![Alt Text](/Results/4059_mtl_result.PNG)
![Alt Text](/Results/6960_image.PNG) | ![Alt Text](/Results/6960_gt.PNG) | ![Alt Text](/Results/6960_two.PNG) | ![Alt Text](/Results/6960_mtl_result.PNG)
![Alt Text](/Results/7_image.PNG) | ![Alt Text](/Results/7_gt.PNG) | ![Alt Text](/Results/7_two.PNG) | ![Alt Text](/Results/7_mtl.PNG)


## Acknowledgments
Parts of the code were heavily derived from https://github.com/qubvel/segmentation_models and https://github.com/anilbatra2185/road_connectivity.

I would also like to thank Liangliang Nan and Frido Kuijper for their contribution in this project.
