import keras
import segmentation_models as sm
from soft_clDice_loss import soft_cldice_losses, clDice_Dice
from soft_cldice_second_version import soft_cldice_loss_version2, combined_loss_version2
import numpy as np 
import os
import scipy.misc as misc
import matplotlib.pyplot as plt


def define_loss(loss, class_weights):
    if loss == "cce":
        if class_weights == "yes":
            total_loss = sm.losses.CategoricalCELoss(class_weights = np.array([1, 10]))
        else:
            total_loss = sm.losses.CategoricalCELoss()

    elif loss == 'cce_dice':
        if class_weights == "yes":
            cce = sm.losses.CategoricalCELoss(class_weights = np.array([1, 10]))
            dice_loss = sm.losses.DiceLoss(class_weights = np.array([1, 10])) 
            total_loss = cce + dice_loss
        else:
            total_loss = sm.losses.CategoricalCELoss() + sm.losses.DiceLoss()

    elif loss == 'cce_jaccard':
        total_loss = sm.losses.CategoricalCELoss() + sm.losses.JaccardLoss()

    elif loss == "dice":
        if class_weights == "yes":
            total_loss = sm.losses.DiceLoss(class_weights = np.array([1, 10]))
        else:
            total_loss = sm.losses.DiceLoss()

    elif loss == "clDice":
        total_loss = soft_cldice_losses

    elif loss == "clDice_dice":
        total_loss = clDice_Dice

    elif loss == "clDice2":
        total_loss = soft_cldice_loss_version2(k=5, data_format="channels_last")

    elif loss == "clDice2_dice2":
        total_loss = combined_loss_version2

    elif loss == 'focal':
        total_loss = sm.losses.CategoricalFocalLoss()
    
    elif loss == 'jaccard':
        total_loss = sm.losses.JaccardLoss()
    elif loss == 'jaccard_focal':
        total_loss = sm.losses.JaccardLoss() + (1 * sm.losses.CategoricalFocalLoss())

    elif loss == 'dice_focal':
        if class_weights == "yes":
            dice_loss = sm.losses.DiceLoss(class_weights = np.array([1, 10])) 
            focal_loss = sm.losses.CategoricalFocalLoss()
            total_loss = dice_loss + (1 * focal_loss)
        else:
            total_loss = sm.losses.DiceLoss() + (1 * sm.losses.CategoricalFocalLoss())       

    return total_loss

def define_mtl_loss(loss1, loss2, loss3, loss4, class_weights):
    loss1 = define_loss(loss1, class_weights)
    loss2 = define_loss(loss2, class_weights)
    if loss3 != None:
        loss3 = define_loss(loss3, class_weights)
        if loss4 != None:
            loss4 = define_loss(loss4, class_weights)
            LOSS = {'output_task_1': loss1, 'output_task_2': loss2, 'output_task_3': loss3, 'output_task_4': loss4}
        else:
            LOSS = {'output_task_1': loss1, 'output_task_2': loss2, 'output_task_3': loss3}
    else:
        LOSS = {'output_task_1': loss1, 'output_task_2': loss2}
        
    return LOSS


def define_classes(task):
    if task == 'two' or task == 'centerline' or task == 'intersection':
        CLASSES = ['background', 'road']
    elif task == 'orientation':
        CLASSES = [str(i) for i in list(range(0, 37))]  # orientation
    elif task=='gaussian':
        CLASSES = [str(i) for i in list(range(0, 42))]  # gaussian
    else:
        # if task == None
        CLASSES = None

    return CLASSES

def get_n_classes(task):
    if task == 'two' or task == 'intersection' or task == 'centerline':
        n_classes = 2
    elif task == 'orientation':
        n_classes = 37
    else:
        n_classes = 42
    
    return n_classes

def get_n_tasks(task1, task2, task3, task4):
    if task4 == None:
        if task3 == None:
            n_tasks = 2
        else:
            n_tasks = 3
    else:
        n_tasks = 4

    return n_tasks


def define_metrics(n_tasks):
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    if n_tasks==2:
        METRICS = {'output_task_1': metrics, 'output_task_2': metrics}
    elif n_tasks==3:
        METRICS = {'output_task_1': metrics, 'output_task_2': metrics, 'output_task_3': metrics}
    elif n_tasks==4:
        METRICS = {'output_task_1': metrics, 'output_task_2': metrics, 'output_task_3': metrics, 'output_task_4':metrics}
    return METRICS

def define_metrics_for_mtl_evaluation(n_tasks):
    m = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), sm.metrics.Precision(threshold=0.5), sm.metrics.Recall(threshold=0.5)]
    if n_tasks==2:
        METRICS = {'output_task_1': m, 'output_task_2': m}
    elif n_tasks==3:
        METRICS = {'output_task_1': m, 'output_task_2': m, 'output_task_3': m}
    elif n_tasks==4:
        METRICS = {'output_task_1': m, 'output_task_2': m, 'output_task_3': m, 'output_task_4':m}
    return METRICS

def define_directory_of_data(base_dir, data_name='images', status='train', os='windows'):

    if os=='windows':
        if data_name=='images':
            ending = '\\{}\\images'.format(status)
        elif data_name=='two':
            ending = '\\{}\\2m_road_gt'.format(status)
        elif data_name=='orientation':
            ending = '\\{}\\orientation_gt'.format(status)
        elif data_name=='centerline':
            ending = '\\{}\\centerline_gt'.format(status)
        elif data_name=='intersection':
            ending = '\\{}\\intersection_gt'.format(status)
        elif data_name=='gaussian':
            ending = '\\{}\\gaussian_gt'.format(status)
        else:
            # data_name==None:
            return None

    else:
        # For Linux 
        if data_name=='images':
            ending = '/{}/images'.format(status)
        elif data_name=='two':
            ending = '/{}/2m_road_gt'.format(status)
        elif data_name=='orientation':
            ending = '/{}/orientation_gt'.format(status)
        elif data_name=='centerline':
            ending = '/{}/centerline_gt'.format(status)
        elif data_name=='intersection':
            ending = '/{}/intersection_gt'.format(status)
        elif data_name=='gaussian':
            ending = '/{}/gaussian_gt'.format(status)
        else:
            # data_name==None:
            return None

    dir = base_dir + ending
    return dir

def find_region_files(d, s): # Wrap this up as a nice function with a docstring.
    "Returns list of files in directory d which have the string s"
    files = os.listdir(d) # Use better names than "List"
    matched_files = []    # List to hold matched file names
    for f in files:
        if s in f:
            # full_name = os.path.join(d, f) # Get full name to the file in question
            matched_files.append(f)
    return matched_files # Return a list of matched files


def save_img_for_comparison(fname, **images):
    'Save images '
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig((fname+'.pdf'))

##Calculate Intersection Over Union Score for predicted layer

def GetIOU(Pred,GT,NumClasses,ClassNames=[], DisplyResults=False): #Given A ground true and predicted labels return the intersection over union for each class
    # and the union for each class
    ClassIOU=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses): # Go over all classes
        Intersection=np.float32(np.sum((Pred==GT)*(GT==i)))# Calculate intersection
        Union=np.sum(GT==i)+np.sum(Pred==i)-Intersection # Calculate Union
        if Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union
            pred_accuracy = np.float32(np.sum(Pred == GT)) / GT.size

    #------------Display results-------------------------------------------------------------------------------------
    if DisplyResults:
       for i in range(len(ClassNames)):
            print(ClassNames[i]+") "+str(ClassIOU[i]))
       print("Mean Classes IOU) "+str(np.mean(ClassIOU)))
       print("Image Predicition Accuracy)" + str(np.float32(np.sum(Pred == GT)) / GT.size))
    #-------------------------------------------------------------------------------------------------

    return ClassIOU, ClassWeight, pred_accuracy

def GetIOU_F1_P_R(Pred,GT,NumClasses,ClassNames=[], DisplyResults=False): #Given A ground true and predicted labels return the intersection over union for each class

    # and the union for each class
    ClassIOU=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassF1=np.zeros(NumClasses)#Vector that Contain F1 per class
    ClassP=np.zeros(NumClasses)#Vector that Contain Precision per class
    ClassR=np.zeros(NumClasses)#Vector that Contain Recall per class
    ClassTP=np.zeros(NumClasses)#Vector that Contain True Positive per class
    ClassTN=np.zeros(NumClasses)#Vector that Contain True Negative per class
    ClassFP=np.zeros(NumClasses)#Vector that Contain False Positive per class
    ClassFN=np.zeros(NumClasses)#Vector that Contain False Negative per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    
    IgnoreValuesMaskR=np.ones(NumClasses, dtype=bool)
    IgnoreValuesMaskP=np.ones(NumClasses, dtype=bool)
    IgnoreValuesMaskF1=np.ones(NumClasses, dtype=bool)
    IgnoreValuesMaskIOU=np.ones(NumClasses, dtype=bool)

    for i in range(NumClasses): # Go over all classes
        Intersection=np.float32(np.sum((Pred==GT)*(GT==i)))# Calculate intersection
        Union=np.sum(GT==i)+np.sum(Pred==i)-Intersection # Calculate Union

        ClassTP[i]=np.float32(np.sum((Pred==i)*(GT==i)))
        ClassTN[i]=np.float32(np.sum((Pred!=i)*(GT!=i)))
        ClassFP[i]=np.float32(np.sum((Pred!=i)*(GT==i)))
        ClassFN[i]=np.float32(np.sum((Pred==i)*(GT!=i)))

        if (ClassTP[i] + ClassFP[i])==0.0:
            ClassP[i]=1
            IgnoreValuesMaskP[i] = False
        else:
            ClassP[i]=(ClassTP[i])/(ClassTP[i] + ClassFP[i])

        if (ClassTP[i] + ClassFN[i])==0.0:
            ClassR[i]=1
            IgnoreValuesMaskR[i] = False
        else:
            ClassR[i]=(ClassTP[i])/(ClassTP[i] + ClassFN[i])

        if (ClassP[i]+ClassR[i]) == 0.0:
            ClassF1[i] = 1
            IgnoreValuesMaskF1[i] = False
        else:
            ClassF1[i]=(2*ClassP[i]*ClassR[i])/(ClassP[i]+ClassR[i])

        if np.sum(GT==i)==0 and np.sum(Pred==i) == 0:
            ClassIOU[i]=1
            ClassWeight[i]=Union
            IgnoreValuesMaskIOU[i]=False
        elif Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union
            # pred_accuracy = np.float32(np.sum(Pred == GT)) / GT.size
            
    #------------Display results-------------------------------------------------------------------------------------
    if DisplyResults:
       for i in range(len(ClassNames)):
            print(ClassNames[i]+") "+str(ClassIOU[i]))
       print("Mean Classes IOU) "+str(np.mean(ClassIOU)))
       print("Image Predicition Accuracy)" + str(np.float32(np.sum(Pred == GT)) / GT.size))
       print("Mean Classes F1: " + str(np.mean(ClassF1)))
    #-------------------------------------------------------------------------------------------------

    mask_arrIOU = np.ma.masked_array(ClassIOU, mask=IgnoreValuesMaskIOU)  
    mask_arrF1 = np.ma.masked_array(ClassF1, mask=IgnoreValuesMaskF1)  
    mask_arrP = np.ma.masked_array(ClassP, mask=IgnoreValuesMaskP)  
    mask_arrR = np.ma.masked_array(ClassR, mask=IgnoreValuesMaskR)  

    IOU = mask_arrIOU.mean() 
    F1 = mask_arrF1.mean()
    P = mask_arrP.mean()
    R = mask_arrR.mean() 

    return IOU, F1, P, R

def get_mtl_predictions(model, n_tasks, img):

    if n_tasks == 2:
        pr1, pr2 = model.predict(img)
        return pr1, pr2, None, None

    elif n_tasks == 3:
        pr1, pr2, pr3 = model.predict(img)
        return  pr1, pr2, pr3, None

    elif n_tasks == 4:
        pr1, pr2, pr3, pr4 = model.predict(img)
        return pr1, pr2, pr3, pr4

def get_mtl_batch(dataset, n_tasks, i):

    if n_tasks == 2:
        image, gt1, gt2 = dataset[i]
        return image, gt1, gt2, None, None

    elif n_tasks == 3:
        image, gt1, gt2, gt3 = dataset[i]
        return image, gt1, gt2, gt3, None

    elif n_tasks == 4:
        image, gt1, gt2, gt3, gt4 = dataset[i]
        return image, gt1, gt2, gt3, gt4

def reverse_mtl_one_hot(gt1, gt2, gt3, gt4, pr1, pr2, pr3, pr4, n_tasks):

    if n_tasks == 2:
        gt1 = np.argmax(gt1, axis=2)
        gt2 = np.argmax(gt2, axis=2)
        pr1 = np.squeeze(pr1, axis=0)
        pr1 = np.argmax(pr1, axis=2)  # pr1.shape= (256, 256)
        pr2 = np.squeeze(pr2, axis=0)
        pr2 = np.argmax(pr2, axis=2)
        return gt1, gt2, None, None, pr1, pr2, None, None

    elif n_tasks == 3:
        gt1 = np.argmax(gt1, axis=2)
        gt2 = np.argmax(gt2, axis=2)
        gt3 = np.argmax(gt3, axis=2)
        pr1 = np.squeeze(pr1, axis=0)
        pr1 = np.argmax(pr1, axis=2)  # pr1.shape= (256, 256)
        pr2 = np.squeeze(pr2, axis=0)
        pr2 = np.argmax(pr2, axis=2)
        pr3 = np.squeeze(pr3, axis=0)
        pr3 = np.argmax(pr3, axis=2)
        return gt1, gt2, gt3, None, pr1, pr2, pr3, None
    
    elif n_tasks == 4:
        gt1 = np.argmax(gt1, axis=2)
        gt2 = np.argmax(gt2, axis=2)
        gt3 = np.argmax(gt3, axis=2)
        gt4 = np.argmax(gt4, axis=2)
        pr1 = np.squeeze(pr1, axis=0)
        pr1 = np.argmax(pr1, axis=2)  # pr1.shape= (256, 256)
        pr2 = np.squeeze(pr2, axis=0)
        pr2 = np.argmax(pr2, axis=2)
        pr3 = np.squeeze(pr3, axis=0)
        pr3 = np.argmax(pr3, axis=2)
        pr4 = np.squeeze(pr4, axis=0)
        pr4 = np.argmax(pr4, axis=2)
        return gt1, gt2, gt3, gt4, pr1, pr2, pr3, pr4

def visualize_MTL(**images):

    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        if image is None:
            continue
        else:
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
    plt.show()

def combine_predictions(pr1, pr2, pr3, pr4, task2, task3, task4):
    pr=np.zeros((pr1.shape))
    pr[pr1==1]=1

    if task2 == 'intersection' or task2 == 'centerline':
        pr[pr2==1]=1
    elif task2 == 'gaussian':
        pr[pr2>41]=1
    else:
        pr[pr2<36]=1

    if task3 is not None:
        if task3 == 'intersection' or task3 == 'centerline':
            pr[pr3==1]=1
        elif task3 == 'gaussian':
            pr[pr3>41]=1
        else:
            pr[pr3<36]=1

    if task4 is not None:
        if task4 == 'orientation':
            pr[pr4<36]=1
        elif task4 == 'gaussian':
            pr[pr4>41]=1
        
    return pr