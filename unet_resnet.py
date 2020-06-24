import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import keras
from keras import applications
from keras.utils import multi_gpu_model
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
import albumentations as A
import argparse
from helpers import define_loss, define_classes, get_n_classes, define_metrics, define_directory_of_data, get_n_tasks, find_region_files
from cldice_metric import clDice
import random

# Receive input before the training starts
parser = argparse.ArgumentParser()

parser.add_argument("--task",
    required=True,
    type=str,
    help="Name the task: centerline, gaussian, intersection, orientation, two")

parser.add_argument("--loss",
    required=True, 
    type=str,
    help="Provide loss function- cce, dice, dice_focal, clDice, clDice_dice, clDice2, clDice2, dice2")

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
    help="Which region do you want to train? Choises: None, Vegas, Paris, Shanghai or Khartoum",
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

    CLASSES = define_classes(args.task)
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            region=None
    ):
        if region==None:
            self.ids = os.listdir(images_dir)
        else:
            self.ids = find_region_files(images_dir, region)

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        if args.task == 'two' or args.task == 'intersection' or args.task == 'centerline':
            mask = np.where(mask==255, 1, mask)  # mask had only 2 values 0 and 255, we convert 255 to 1
            # visualize(
            #     img=image,
            #     mask=mask
            # )
        elif args.task == 'gaussian':
            count = 0
            for boundary in range(0,256,6):
                a = boundary
                b = boundary + 7
                mask[(mask>a)&(mask<b)] = count
                count += 1
            # visualize(
            #     img=image,
            #     mask=mask
            # )
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # # uncomment if you need to add background
        # # add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
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


# Define Data Directory
DATA_DIR = "C:\\Users\\folder\\of\\data"
# example DATA_DIR = "C:\\SpaceNet3-prepared"

# Define some parameters
BACKBONE = 'resnet34'
BATCH_SIZE = 15
INPUT_SHAPE = (256,256,3)
LEARNING_RATE = 0.001
EPOCHS = 50

CLASSES = define_classes(args.task)

if args.encoder_weights == "yes":
    ENCODER_WEIGHTS = 'imagenet'
else:
    ENCODER_WEIGHTS = None

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
activation = 'softmax'
# activation = 'sigmoid'

# Dataset for train images
train_dataset = Dataset(
    images_dir=define_directory_of_data(base_dir=DATA_DIR, data_name='images', status='train', os='windows'), 
    masks_dir=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task, status='train', os='windows'),
    classes=CLASSES, 
    #augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
    region=args.region,
)

# Dataset for validation images
valid_dataset = Dataset(
    images_dir=define_directory_of_data(base_dir=DATA_DIR, data_name='images', status='validation', os='windows'), 
    masks_dir=define_directory_of_data(base_dir=DATA_DIR, data_name=args.task, status='validation', os='windows'),
     classes=CLASSES, 
    #augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
    region=args.region,
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# Check for multiple GPUs
model = sm.Unet(BACKBONE,
                classes=len(CLASSES),
                activation=activation,
                input_shape=INPUT_SHAPE,
                encoder_weights=ENCODER_WEIGHTS
                )

# # Ucomment if you want to plot the model architecture
# import pydot_ng as pydot
# keras.utils.plot_model(model, to_file='model.png')

# Define optimizer
optimizer = keras.optimizers.Adam(LEARNING_RATE)

# Define Loss function
total_loss = define_loss(args.loss, args.class_weights)

# Define which metrics will evaluate your model
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined op
model.compile(optimizer, total_loss, metrics)

# # Uncomment to visualize some examples from both the train and validation datasets
# images = random.sample(range(1000), 5)
# for i in images:
    # image, mask = train_dataset[i] # get some sample
    # visualize(
    #     image=image,
    #     background_mask=mask[..., 0].squeeze(),
    #     road_mask=mask[..., 1].squeeze()
    # )

    # image, mask = valid_dataset[i] # get some sample
    # visualize(
    #     image=image,
    #     background_mask=mask[..., 0].squeeze(),
    #     road_mask=mask[..., 1].squeeze()
    # )

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 256, 256, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 256, 256, len(CLASSES))
assert valid_dataloader[0][0].shape == (1, 256, 256, 3)

# define callbacks for learning rate scheduling and best checkpoints saving
best_model_name = './best_model_task_{}_loss_{}_encoder_weights_{}_class_weights_{}_region_{}.h5'.format(args.task, args.loss, args.encoder_weights, args.class_weights, args.region)
callbacks = [
    keras.callbacks.ModelCheckpoint(best_model_name, save_weights_only=True, save_best_only=True, mode='min'),
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

fname = 'PLOT_task_{}_loss_{}_encoder_weights_{}_class_weights_{}_region_{}.pdf'.format(args.task, args.loss, args.encoder_weights, args.class_weights, args.region)
plt.savefig(fname)  


### Model Evaluation
print("Evaluation started...")
# If you want to load weights and not re-train, use the following lines of code
# weight_folder = "C:\\path\\to\\weight\\folder\\"
# weight_file = "weight_file.h5"
# fname = weight_folder + weight_file
# model.load_weights(fname, by_name=True)

test_dataset = Dataset(
    images_dir=define_directory_of_data(DATA_DIR, data_name='images', status='test', os='windows'), 
    masks_dir=define_directory_of_data(DATA_DIR, data_name=args.task, status='test', os='windows'), 
    classes=CLASSES, 
    #augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
    region=args.region
)
test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

scores = model.evaluate_generator(test_dataloader)
for i in range(len(scores)):
    print(model.metrics_names[i], '= ', scores[i])

print('Calculating clDice metric...')
total_cldice_score = 0.0
total_n_img_ignore = 0
for i in range(len(test_dataset)):
    image, gt = test_dataset[i] # gt.shape = (256, 256, 37)
    img = np.expand_dims(image, axis=0)     # image.shape = (1, 256, 256, 3)
    pr = model.predict(img)     # pr.shape = (1, 256, 256, 37)

    # Now let's reverse one hot encoding
    gt = np.argmax(gt, axis=2)  #  gt.shape= (256, 256)
    pr = np.squeeze(pr, axis=0)
    pr = np.argmax(pr, axis=2)  # pr.shape= (256, 256)

    # visualize(
    #     image=image,
    #     gt=gt,
    #     pr=pr
    #     )

    #compute the clDice metric
    clDice_score, images_to_ignore = clDice(pr, gt)
    # print("For image {} the clDice score is: {}".format(pair, clDice_score))
    total_cldice_score += clDice_score
    total_n_img_ignore += images_to_ignore

print("Mean clDice Score: ", total_cldice_score / (len(test_dataset)-total_n_img_ignore))
        
print('Evaluation finished\n')


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

print('Let\'s visualize some predictions!')
# define the len of the validation dataset within the range parameter
prediction_images = random.sample(range(18731), 50)
path_for_img = 'C:\\Users\\kaniourasp\\Desktop\\new_examples\\'
for i in prediction_images:
    print("Picture selected is: ", i)
    image, gt = test_dataset[i] # gt.shape = (256, 256, 37)
    img = np.expand_dims(image, axis=0)     # image.shape = (1, 256, 256, 3)
    pr = model.predict(img)     # pr.shape = (1, 256, 256, 37)

    # Now let's reverse one hot encoding
    gt = np.argmax(gt, axis=2)  #  gt.shape= (256, 256)
    pr = np.squeeze(pr, axis=0)
    pr = np.argmax(pr, axis=2)  # pr.shape= (256, 256)
    # img = np.squeeze(img, axis=0)  # img.shape =  (256, 256, 3)

    # # If you want to Save images, uncomment the following lines of code
    # fname = path_for_img + 'single_' + str(i)
    # save_img_for_comparison(fname,
    #     image=image,
    #     gt=gt,
    #     pr=pr
    #     )

    visualize(
        image=image,
        gt=gt,
        pr=pr
        )