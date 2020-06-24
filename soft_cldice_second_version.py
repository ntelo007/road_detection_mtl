import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K


def dice_loss(data_format="channels_first"):
    """dice loss function for tensorflow/keras
        calculate dice loss per batch and channel of each sample.
    Args:
        data_format: either channels_first or channels_last
    Returns:
        loss_function(y_true, y_pred)  
    """

    def loss(target, pred):
        if data_format == "channels_last":
            pred = tf.transpose(pred, (0, 3, 1, 2))
            target = tf.transpose(target, (0, 3, 1, 2))
        
        smooth = 1.0
        iflat = tf.reshape(
            pred, (tf.shape(pred)[0], tf.shape(pred)[1], -1)
        )  # batch, channel, -1
        tflat = tf.reshape(target, (tf.shape(target)[0], tf.shape(target)[1], -1))
        intersection = K.sum(iflat * tflat, axis=-1)
        return 1 - ((2.0 * intersection + smooth)) / (
            K.sum(iflat, axis=-1) + K.sum(tflat, axis=-1) + smooth
        )

    return loss


def soft_skeletonize(x, thresh_width=10):
    """
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - needs to be greater then or equal to the maximum radius for the tube-like structure
    """

    minpool = (
        lambda y: K.pool2d(
            y * -1,
            pool_size=(3, 3),
            strides=(1, 1),
            pool_mode="max",
            data_format="channels_first",
            padding="same",
        )
        * -1
    )
    maxpool = lambda y: K.pool2d(
        y,
        pool_size=(3, 3),
        strides=(1, 1),
        pool_mode="max",
        data_format="channels_first",
        padding="same",
    )

    for i in range(thresh_width):
        min_pool_x = minpool(x)
        contour = K.relu(maxpool(min_pool_x) - min_pool_x)
        x = K.relu(x - contour)
    return x


def norm_intersection(center_line, vessel):
    """
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    """
    smooth = 1.0
    clf = tf.reshape(
        center_line, (tf.shape(center_line)[0], tf.shape(center_line)[1], -1)
    )
    vf = tf.reshape(vessel, (tf.shape(vessel)[0], tf.shape(vessel)[1], -1))
    intersection = K.sum(clf * vf, axis=-1)
    return (intersection + smooth) / (K.sum(clf, axis=-1) + smooth)


def soft_cldice_loss_version2(k=10, data_format="channels_first"):
    """clDice loss function for tensorflow/keras
    Args:
        k: needs to be greater or equal to the maximum radius of the tube structure.
        data_format: either channels_first or channels_last        
    Returns:
        loss_function(y_true, y_pred)  
    """

    def loss(target, pred):
        if data_format == "channels_last":
            pred = tf.transpose(pred, (0, 3, 1, 2))
            target = tf.transpose(target, (0, 3, 1, 2))

        cl_pred = soft_skeletonize(pred, thresh_width=k)
        target_skeleton = soft_skeletonize(target, thresh_width=k)
        iflat = norm_intersection(cl_pred, target)
        tflat = norm_intersection(target_skeleton, pred)
        intersection = iflat * tflat
        return 1 - ((2.0 * intersection) / (iflat + tflat))

    return loss

    # Or combine dice + cldice similiar to the experiments in the paper
def combined_loss_version2(y_true, y_pred):
    alpha = 0.5
    data_format="channels_last"
    return (alpha * dice_loss(data_format=data_format)(y_true, y_pred) + 
            (1-alpha) * soft_cldice_loss_version2(k=5, data_format=data_format)(y_true, y_pred))

# use example
# from dice_helpers_tf import dice_loss, soft_cldice_loss
# cldice_loss = soft_cldice_loss(k=5, data_format="channels_last")
# model.compile(loss=cldice_loss, [...])
# or
# model.compile(loss= combined_loss, [...])
