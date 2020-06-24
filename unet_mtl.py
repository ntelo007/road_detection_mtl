import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import keras
from keras import applications
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
import albumentations as A
import argparse
from helpers2 import define_mtl_loss, define_classes, get_n_classes, define_metrics, define_directory_of_data, get_n_tasks, find_region_files, get_mtl_batch, reverse_mtl_one_hot, visualize_MTL
import random
from cldice_metric import clDice


# Receive input before the training starts
parser = argparse.ArgumentParser()

parser.add_argument("--split", type=str, help="Early or late split? acceptable ansers: early-late", default='early')

parser.add_argument("--task1", required=True, type=str, help="Provide the name of task 1: for now only two is acceptable")
parser.add_argument("--task2", required=True, type=str, help="Provide the name of task 2: centerline, intersection, gaussian or orientation")
parser.add_argument("--task3", type=str, help="Provide the name of task 3: None, intersection, orientation, gaussian or centerline", default=None)
parser.add_argument("--task4",  type=str, help="Provide the name of task 4: None, intersection, orientation, gaussian or centerline", default=None)

parser.add_argument("--loss1", required=True, type=str, help="Provide loss function for task1: cce, cce_jaccard, cce_dice, dice, dice_focal, clDice_dice or clDice")
parser.add_argument("--loss2", required=True, type=str, help="Provide loss function for task2: cce, cce_jaccard, cce_dice, dice, dice_focal, clDice_dice or clDice")
parser.add_argument("--loss3", type=str, help="Provide loss function for task3: None, cce, cce_jaccard, cce_dice, dice, dice_focal, clDice_dice or clDice", default=None)
parser.add_argument("--loss4", type=str, help="Provide loss function for task4: None, cce, cce_jaccard, cce_dice, dice, dice_focal, clDice_dice or clDice", default=None)

parser.add_argument(
    "--encoder_weights",
    type=str,
    help="Would you like to use pretrained weights of imagenet? choices:yes-no",
    default='yes'
    )

parser.add_argument(
    "--class_weights",
    type=str,
    help="Would you like to use 1-10 ratio as class weights in the loss function? choices:yes-no",
    default='no'
    )

parser.add_argument(
    "--region",
    type=str,
    help="Which region do you want to train? Choises: Vegas or Paris or Shanghai or Khartoum",
    default=None
    )

parser.add_argument(
    "--loss_weights",
    type=int,
    help="What weight do you want to apply to task 1? Choices: 1-2-5-10-20",
    default=None
    ) 

args = parser.parse_args()

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


class Dataset:
    """SpaceNet-prepared Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)
    
    """
    
    CLASSES_task_1 = define_classes(args.task1)
    CLASSES_task_2 = define_classes(args.task2)
    CLASSES_task_3 = define_classes(args.task3)
    CLASSES_task_4 = define_classes(args.task4)

    def __init__(
            self, 
            images_dir, 
            masks_dir_1,
            masks_dir_2,
            masks_dir_3,
            masks_dir_4,
            classes_1=CLASSES_task_1,
            classes_2=CLASSES_task_2, 
            classes_3=CLASSES_task_3,
            classes_4=CLASSES_task_4,
            augmentation=None, 
            preprocessing=None,
            n_tasks=3,
            region=None,
    ):
        if region==None:
            self.ids = os.listdir(images_dir)
        else:
            self.ids = find_region_files(images_dir, region)

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps_task_1 = [os.path.join(masks_dir_1, image_id) for image_id in self.ids]
        self.masks_fps_task_2 = [os.path.join(masks_dir_2, image_id) for image_id in self.ids]
        if n_tasks==3:
            if masks_dir_3 == None:
                self.masks_fps_task_3 = None
            else:
                self.masks_fps_task_3 = [os.path.join(masks_dir_3, image_id) for image_id in self.ids]
        elif n_tasks==4:
            self.masks_fps_task_3 = [os.path.join(masks_dir_3, image_id) for image_id in self.ids]
            if masks_dir_1 == None:
                self.masks_fps_task_4 = None
            else:
                self.masks_fps_task_4 = [os.path.join(masks_dir_4, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values_task_1 = [self.CLASSES_task_1.index(cls.lower()) for cls in classes_1]
        self.class_values_task_2 = [self.CLASSES_task_2.index(cls.lower()) for cls in classes_2]
        if n_tasks==3:
            self.class_values_task_3 = [self.CLASSES_task_3.index(cls.lower()) for cls in classes_3]
        elif n_tasks==4:
            self.class_values_task_3 = [self.CLASSES_task_3.index(cls.lower()) for cls in classes_3]
            self.class_values_task_4 = [self.CLASSES_task_4.index(cls.lower()) for cls in classes_4]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_from_task_1 = cv2.imread(self.masks_fps_task_1[i], 0)
        if args.task1 == 'two' or args.task1 == 'intersection' or args.task1 == 'centerline':
            mask_from_task_1 = np.where(mask_from_task_1==255, 1, mask_from_task_1)  # mask had only 2 values 0 and 255, we convert 255 to 1

        mask_from_task_2 = cv2.imread(self.masks_fps_task_2[i], 0)
        if args.task2 == 'two' or args.task2 == 'intersection' or args.task2 == 'centerline':
            mask_from_task_2 = np.where(mask_from_task_2==255, 1, mask_from_task_2)  # mask had only 2 values 0 and 255, we convert 255 to 1
        elif args.task2 == 'gaussian':
            count = 0
            for boundary in range(0,256,6):
                a = boundary
                b = boundary + 7
                mask_from_task_2[(mask_from_task_2>a)&(mask_from_task_2<b)] = count
                count += 1

        if args.task3 != None:
            mask_from_task_3 = cv2.imread(self.masks_fps_task_3[i], 0)
            if args.task3 == 'two' or args.task3 == 'intersection' or args.task3 == 'centerline':
                mask_from_task_3 = np.where(mask_from_task_3==255, 1, mask_from_task_3)  # mask had only 2 values 0 and 255, we convert 255 to 1
            elif args.task3 == 'gaussian':
                count = 0
                for boundary in range(0,256,6):
                    a = boundary
                    b = boundary + 7
                    mask_from_task_3[(mask_from_task_3>a)&(mask_from_task_3<b)] = count
                    count += 1
         
        if args.task4 != None:
            mask_from_task_4 = cv2.imread(self.masks_fps_task_4[i], 0)
            if args.task4 == 'gaussian':
                count = 0
                for boundary in range(0,256,6):
                    a = boundary
                    b = boundary + 7
                    mask_from_task_4[(mask_from_task_4>a)&(mask_from_task_4<b)] = count
                    count += 1


            
        # extract certain classes from mask (e.g. cars)
        masks_task_1 = [(mask_from_task_1 == v) for v in self.class_values_task_1]
        mask_from_task_1 = np.stack(masks_task_1, axis=-1).astype('float')

        masks_task_2 = [(mask_from_task_2 == v) for v in self.class_values_task_2]
        mask_from_task_2 = np.stack(masks_task_2, axis=-1).astype('float')

        if args.task3 != None:
            masks_task_3 = [(mask_from_task_3 == v) for v in self.class_values_task_3]
            mask_from_task_3 = np.stack(masks_task_3, axis=-1).astype('float')
        if args.task4 != None:
            masks_task_4 = [(mask_from_task_4 == v) for v in self.class_values_task_4]
            mask_from_task_4 = np.stack(masks_task_4, axis=-1).astype('float')
        
        # # add background if mask is not binary
        # if mask_from_task_1.shape[-1] != 1:
        #     background = 1 - mask_from_task_1.sum(axis=-1, keepdims=True)
        #     mask_from_task_1 = np.concatenate((mask_from_task_1, background), axis=-1)
        # if mask_from_task_2.shape[-1] != 1:
        #     background = 1 - mask_from_task_2.sum(axis=-1, keepdims=True)
        #     mask_from_task_2 = np.concatenate((mask_from_task_2, background), axis=-1)
        # if mask_from_task_3.shape[-1] != 1:
        #     background = 1 - mask_from_task_3.sum(axis=-1, keepdims=True)
        #     mask_from_task_3 = np.concatenate((mask_from_task_3, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            if n_tasks == 2:
                sample = self.augmentation(image=image, mask_1=mask_from_task_1, mask_2=mask_from_task_2)
                image, mask_from_task_1, mask_from_task_2= sample['image'], sample['mask_1'], sample['mask_2']
            elif n_tasks == 3:
                sample = self.augmentation(image=image, mask_1=mask_from_task_1, mask_2=mask_from_task_2, mask_3=mask_from_task_3)
                image, mask_from_task_1, mask_from_task_2, mask_from_task_3 = sample['image'], sample['mask_1'], sample['mask_2'], sample['mask_3']
            else:
                sample = self.augmentation(image=image, mask_1=mask_from_task_1, mask_2=mask_from_task_2, mask_3=mask_from_task_3, mask_4=mask_from_task_4)
                image, mask_from_task_1, mask_from_task_2, mask_from_task_3, mask_from_task_4 = sample['image'], sample['mask_1'], sample['mask_2'], sample['mask_3'], sample['mask_4']

        # apply preprocessing
        if self.preprocessing:
            if n_tasks == 2:
                sample = self.preprocessing(image=image, mask_1=mask_from_task_1, mask_2=mask_from_task_2)
                image, mask_from_task_1, mask_from_task_2 = sample['image'], sample['mask_1'], sample['mask_2']
            elif n_tasks == 3:
                sample = self.preprocessing(image=image, mask_1=mask_from_task_1, mask_2=mask_from_task_2, mask_3=mask_from_task_3)
                image, mask_from_task_1, mask_from_task_2, mask_from_task_3 = sample['image'], sample['mask_1'], sample['mask_2'], sample['mask_3']
            else:
                sample = self.preprocessing(image=image, mask_1=mask_from_task_1, mask_2=mask_from_task_2, mask_3=mask_from_task_3, mask_4=mask_from_task_4)
                image, mask_from_task_1, mask_from_task_2, mask_from_task_3, mask_from_task_4 = sample['image'], sample['mask_1'], sample['mask_2'], sample['mask_3'], sample['mask_4']
        
        # return image, mask_from_task_1[:, :, 0], mask_from_task_2[:, :, 0], mask_from_task_3[:, :, 0]
        if n_tasks==2:
            return image, mask_from_task_1, mask_from_task_2
        elif n_tasks==3:
            return image, mask_from_task_1, mask_from_task_2, mask_from_task_3
        else:
            return image, mask_from_task_1, mask_from_task_2, mask_from_task_3, mask_from_task_4
        

        
    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False, n_tasks=3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size

        # numpy arrays
        n_classes_task_1 = get_n_classes(args.task1)
        n_classes_task_2 = get_n_classes(args.task2)
        X = np.zeros((self.batch_size, 256, 256, 3))
        Y_1 = np.zeros((self.batch_size, 256, 256, n_classes_task_1))
        Y_2 = np.zeros((self.batch_size, 256, 256, n_classes_task_2))
        index = 0

        if n_tasks == 2:
            for j in range(start, stop):
                X[index, :, :, :] = self.dataset[j][0]
                Y_1[index, :, :, :] = self.dataset[j][1]
                Y_2[index, :, :, :] = self.dataset[j][2]
                index += 1
            batch = (X, [Y_1, Y_2])
            return batch

        elif n_tasks == 3:
            n_classes_task_3 = get_n_classes(args.task3)
            Y_3 = np.zeros((self.batch_size, 256, 256, n_classes_task_3))
            for j in range(start, stop):
                X[index, :, :, :] = self.dataset[j][0]
                Y_1[index, :, :, :] = self.dataset[j][1]
                Y_2[index, :, :, :] = self.dataset[j][2]
                Y_3[index, :, :, :] = self.dataset[j][3]
                index += 1
            batch = (X, [Y_1, Y_2, Y_3])
            return batch
        
        else:
            n_classes_task_3 = get_n_classes(args.task3)
            Y_3 = np.zeros((self.batch_size, 256, 256, n_classes_task_3))
            n_classes_task_4 = get_n_classes(args.task4)
            Y_4 = np.zeros((self.batch_size, 256, 256, n_classes_task_4))
            for j in range(start, stop):
                X[index, :, :, :] = self.dataset[j][0]
                Y_1[index, :, :, :] = self.dataset[j][1]
                Y_2[index, :, :, :] = self.dataset[j][2]
                Y_3[index, :, :, :] = self.dataset[j][3]
                Y_4[index, :, :, :] = self.dataset[j][4]
                index += 1
            batch = (X, [Y_1, Y_2, Y_3, Y_4])
            return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1)
        ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(256, 256)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


# Define some parameters
n_tasks = get_n_tasks(args.task1, args.task2, args.task3, args.task4)

# Define Data Directory
DATA_DIR = "C:\\Users\\folder\\of\\data"
# example DATA_DIR = "C:\\SpaceNet3-prepared"

evaluate = True
BACKBONE = 'resnet34'
BATCH_SIZE = 15
INPUT_SHAPE = (256,256,3)
LEARNING_RATE = 0.001
EPOCHS = 50

CLASSES_task_1 = define_classes(args.task1)
CLASSES_task_2 = define_classes(args.task2)
CLASSES_task_3 = define_classes(args.task3)
CLASSES_task_4 = define_classes(args.task4)


if args.encoder_weights == "yes":
    ENCODER_WEIGHTS = 'imagenet'
else:
    ENCODER_WEIGHTS = None

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
activation = 'softmax'

if args.split == 'late':
    model = sm.Unet_Late_MTL(BACKBONE,
                    activation=activation,
                    input_shape=INPUT_SHAPE,
                    encoder_weights=ENCODER_WEIGHTS,
                    task1=args.task1,
                    task2=args.task2,
                    task3=args.task3,
                    task4=args.task4,
                    )
else:
    model = sm.Unet_MTL(BACKBONE,
                    activation=activation,
                    input_shape=INPUT_SHAPE,
                    encoder_weights=ENCODER_WEIGHTS,
                    task1=args.task1,
                    task2=args.task2,
                    task3=args.task3,
                    task4=args.task4,
                    )

# # Uncomment if you want to print the model summary
# print(model.summary())

# # Ucomment if you want to plot the model architecture
# import pydot_ng as pydot
# keras.utils.plot_model(model, to_file='model.png')

# Define optimizer
optimizer = keras.optimizers.Adam(LEARNING_RATE)

# Define Loss function
LOSS = define_mtl_loss(args.loss1, args.loss2, args.loss3, args.loss4, args.class_weights)

# Define which metrics will evaluate your model
METRICS = define_metrics(n_tasks)

# compile keras model with defined optimozer, loss and metrics
if args.loss_weights != None:
    loss_weights = [1]*n_tasks
    loss_weights[0] = args.loss_weights
    model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS, loss_weights=loss_weights)
else:
    model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)

# Dataset for train images
train_dataset = Dataset(
    images_dir=define_directory_of_data(base_dir=DATA_DIR, data_name='images', status='train', os='windows'), 
    masks_dir_1=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task1, status='train', os='windows'),
    masks_dir_2=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task2, status='train', os='windows'),
    masks_dir_3=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task3, status='train', os='windows'),
    masks_dir_4=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task4, status='train', os='windows'), 
    #augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
    n_tasks=n_tasks,
    region=args.region,
)

# Dataset for validation images
valid_dataset = Dataset(
    images_dir=define_directory_of_data(base_dir=DATA_DIR, data_name='images', status='validation', os='windows'), 
    masks_dir_1=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task1, status='validation', os='windows'),
    masks_dir_2=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task2, status='validation', os='windows'),
    masks_dir_3=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task3, status='validation', os='windows'),
    masks_dir_4=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task4, status='validation', os='windows'), 
    #augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
    n_tasks=n_tasks,
    region=args.region,
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)


# # visualize n examples from the train dataset
# n = 10
# ids = np.random.choice(np.arange(len(train_dataset)), size=n)

# for i in ids:
#     image, gt1, gt2, gt3, gt4 = get_mtl_batch(train_dataset, n_tasks, i) # gt.shape = (256, 256, 37)

#     # Now let's reverse one hot encoding
#     gt1 = np.argmax(gt1, axis=2)
#     gt2 = np.argmax(gt2, axis=2)
#     if gt3 is not None:
#         gt3 = np.argmax(gt3, axis=2)
#     if gt4 is not None:
#         gt4 = np.argmax(gt4, axis=2)

#     visualize_MTL(
#         image=image,
#         gt1=gt1,
#         gt2=gt2,
#         gt3=gt3,
#         gt4=gt4,
#     )


# define callbacks for learning rate scheduling and best checkpoints saving
model_name = './best_model_MTL_t1_{}_t2_{}_t3_{}_t4_{}_loss1_{}_loss2_{}_loss3_{}_loss4_{}_encoderWeights_{}_classWeights_{}_region_{}_lossWeights_{}.h5'.format(args.task1, args.task2, args.task3, args.task4, args.loss1, args.loss2, args.loss3, args.loss4, args.encoder_weights, args.class_weights, args.region, args.loss_weights)
callbacks = [
    keras.callbacks.ModelCheckpoint(model_name, save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(patience=2)
]

# train model
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)

# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plot_name = 'PLOT_MTL_t1_{}_t2_{}_t3_{}_t4_{}_loss1_{}_loss2_{}_loss3_{}_loss4_{}_encoderWeights_{}_classWeights_{}_region_{}_lossWeights_{}.pdf'.format(args.task1, args.task2, args.task3, args.task4, args.loss1, args.loss2, args.loss3, args.loss4, args.encoder_weights, args.class_weights, args.region, args.loss_weights)
plt.savefig(plot_name)



### Model Evaluation
if evaluate == True:
    print("Evaluation started...")

    # Dataset to test model
    test_dataset = Dataset(
        images_dir=define_directory_of_data(base_dir=DATA_DIR, data_name='images', status='test', os='windows'), 
        masks_dir_1=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task1, status='test', os='windows'),
        masks_dir_2=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task2, status='test', os='windows'),
        masks_dir_3=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task3, status='test', os='windows'),
        masks_dir_4=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task4, status='test', os='windows'), 
        #augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
        n_tasks=n_tasks,
        region=args.region,
    )

    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    # If you want to load weights and not re-train, use the following lines of code
    # weight_folder = "C:\\path\\to\\weight\\folder\\"
    # weight_file = "weight_file.h5"
    # fname = weight_folder + weight_file
    # model.load_weights(fname, by_name=True)


    # Calculate and print metrics
    scores = model.evaluate_generator(test_dataloader)
    for i in range(len(scores)):
        print(model.metrics_names[i], '= ', scores[i])

    if args.task1 == 'two' or args.task1 == 'centerline':
        print('Calculating clDice metric for task 1...')
        total_cldice_score = 0.0
        total_n_img_ignore = 0
        if combine_tasks == True:
            for i in range(len(test_dataset)):
                image, gt1, gt2, gt3, gt4 = get_mtl_batch(test_dataset, n_tasks, i) # gt.shape = (256, 256, 37)
                img = np.expand_dims(image, axis=0)   # image.shape = (1, 256, 256, 3)
                pr1, pr2, pr3, pr4 = get_mtl_predictions(model, n_tasks, img)#.round() # pr.shape = (1, 256, 256, 37)

                # Now let's reverse one hot encoding
                gt1, gt2, gt3, gt4, pr1, pr2, pr3, pr4 = reverse_mtl_one_hot(gt1, gt2, gt3, gt4, pr1, pr2, pr3, pr4, n_tasks) 

                # Let's combine outputs into one image
                pr = combine_predictions(pr1, pr2, pr3, pr4, task2, task3, task4)

                # visualize(
                #     image=image,
                #     gt1=gt1,
                #     pr=pr
                #     )

                # visualize_MTL(
                #     image=image,
                #     gt1=gt1,
                #     gt2=gt2,
                #     gt3=gt3,
                #     gt4=gt4,
                #     pr1=pr1,
                #     pr2=pr2,
                #     pr3=pr3,
                #     pr4=pr4
                # )

                #compute the clDice metric
                clDice_score, images_to_ignore = clDice(pr, gt1)
                # print("For image {} the clDice score is: {}".format(pair, clDice_score))
                total_cldice_score += clDice_score
                total_n_img_ignore += images_to_ignore
            print("Mean clDice Score without combining outputs of tasks is : ", total_cldice_score / (len(test_dataset)-total_n_img_ignore))
        
        else:
            
            for i in range(len(test_dataset)):
                image, gt1, gt2, gt3, gt4 = get_mtl_batch(test_dataset, n_tasks, i) # gt.shape = (256, 256, 37)
                img = np.expand_dims(image, axis=0)   # image.shape = (1, 256, 256, 3)
                pr1, pr2, pr3, pr4 = get_mtl_predictions(model, n_tasks, img)#.round() # pr.shape = (1, 256, 256, 37)

                # Now let's reverse one hot encoding
                gt1, gt2, gt3, gt4, pr1, pr2, pr3, pr4 = reverse_mtl_one_hot(gt1, gt2, gt3, gt4, pr1, pr2, pr3, pr4, n_tasks) 

                # visualize(
                #     image=image,
                #     gt=gt,
                #     pr=pr
                #     )

                #compute the clDice metric
                clDice_score, images_to_ignore = clDice(pr1, gt1)
                # print("For image {} the clDice score is: {}".format(pair, clDice_score))
                total_cldice_score += clDice_score
                total_n_img_ignore += images_to_ignore
            print("Mean clDice Score without combining outputs of tasks is : ", total_cldice_score / (len(test_dataset)-total_n_img_ignore))
        
    print('Evaluation finished\n')


# Uncomment if you want to visualize predictions
print('Let\'s visualize some predictions for comparison!')
# Define the len of the test dataset within the range parameter
prediction_images = random.sample(range(18731), 50)

def save_img_for_comparison(fname, **images):
    'Save images '
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        # plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig((fname+'.png'))
    plt.close()

if n_tasks == 2:
    for i in prediction_images:
        image, gt1, gt2 = test_dataset[i] # gt.shape = (256, 256, 37)
        img = np.expand_dims(image, axis=0)   # image.shape = (1, 256, 256, 3)
        pr1, pr2= model.predict(img)#.round() # pr.shape = (1, 256, 256, 37)

        # Now let's reverse one hot encoding
        gt1 = np.argmax(gt1, axis=2)  #  gt1.shape= (256, 256)
        gt2 = np.argmax(gt2, axis=2)  #  gt2.shape= (256, 256)

        pr1 = np.squeeze(pr1, axis=0)
        pr1 = np.argmax(pr1, axis=2)  # pr1.shape= (256, 256)
        pr2 = np.squeeze(pr2, axis=0)
        pr2 = np.argmax(pr2, axis=2)  # pr2.shape= (256, 256)

        visualize(
            image=image,
            gt1=gt1,
            gt2=gt2,
            pr1=pr1,
            pr2=pr2,
        )

elif n_tasks == 3:
    for i in prediction_images:
        image, gt1, gt2, gt3 = test_dataset[i] # gt.shape = (256, 256, 37)
        img = np.expand_dims(image, axis=0)   # image.shape = (1, 256, 256, 3)
        pr1, pr2, pr3 = model.predict(img)#.round() # pr.shape = (1, 256, 256, 37)

        # Now let's reverse one hot encoding
        gt1 = np.argmax(gt1, axis=2)  #  gt1.shape= (256, 256)
        gt2 = np.argmax(gt2, axis=2)  #  gt2.shape= (256, 256)
        gt3 = np.argmax(gt3, axis=2)  #  gt3.shape= (256, 256)

        pr1 = np.squeeze(pr1, axis=0)
        pr1 = np.argmax(pr1, axis=2)  # pr1.shape= (256, 256)
        pr2 = np.squeeze(pr2, axis=0)
        pr2 = np.argmax(pr2, axis=2)  # pr2.shape= (256, 256)
        pr3 = np.squeeze(pr3, axis=0)
        pr3 = np.argmax(pr3, axis=2)  # pr3.shape= (256, 256)

        visualize(
            image=image,
            gt1=gt1,
            gt2=gt2,
            gt3=gt3,
            pr1=pr1,
            pr2=pr2,
            pr3=pr3
        )

        # path_for_img = 'C:\\Users\\kaniourasp\\Desktop\\new_examples\\'
        # fname = path_for_img + 'mtl_' + str(i)
        # save_img_for_comparison(fname,
        #     image=image,
        #     gt1=gt1,
        #     #gt2=gt2,
        #     #gt3=gt3,
        #     pr1=pr1,
        #     #pr2=pr2,
        #     #pr3=pr3
        #     )

elif n_tasks == 4:
    for i in prediction_images:
        image, gt1, gt2, gt3, gt4 = test_dataset[i] # gt.shape = (256, 256, 37)
        img = np.expand_dims(image, axis=0)   # image.shape = (1, 256, 256, 3)
        pr1, pr2, pr3, pr4 = model.predict(img)#.round() # pr.shape = (1, 256, 256, 37)

        # Now let's reverse one hot encoding
        gt1 = np.argmax(gt1, axis=2)  #  gt1.shape= (256, 256)
        gt2 = np.argmax(gt2, axis=2)  #  gt2.shape= (256, 256)
        gt3 = np.argmax(gt3, axis=2)  #  gt3.shape= (256, 256)
        gt4 = np.argmax(gt4, axis=2)  #  gt3.shape= (256, 256)

        pr1 = np.squeeze(pr1, axis=0)
        pr1 = np.argmax(pr1, axis=2)  # pr1.shape= (256, 256)
        pr2 = np.squeeze(pr2, axis=0)
        pr2 = np.argmax(pr2, axis=2)  # pr2.shape= (256, 256)
        pr3 = np.squeeze(pr3, axis=0)
        pr3 = np.argmax(pr3, axis=2)  # pr3.shape= (256, 256)
        pr4 = np.squeeze(pr4, axis=0)
        pr4 = np.argmax(pr4, axis=2)  # pr3.shape= (256, 256)

        visualize(
            image=image,
            gt1=gt1,
            gt2=gt2,
            gt3=gt3,
            gt4=gt4,
            pr1=pr1,
            pr2=pr2,
            pr3=pr3,
            pr4=pr4
        )