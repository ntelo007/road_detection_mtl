import tensorflow as tf
import keras.backend as K

def padding_tensor(x, patch_size=3):
    paddings = tf.constant([[0, 0], [int((patch_size - 1) / 2), int((patch_size - 1) / 2)],
                            [int((patch_size - 1) / 2), int((patch_size - 1) / 2)], [0, 0]])
    return tf.pad(x, paddings, "REFLECT")


def norm_intersection(center_line, vessel):
    intersection = tf.reduce_sum(center_line * vessel, axis=(1, 2, 3), keepdims=True)
    return tf.compat.v1.div_no_nan(intersection, tf.reduce_sum(center_line, axis=(1, 2, 3), keepdims=True))


def dilation2d(x, kernel_size=3, dilations=1, strides=1):
    weight = 1. / (kernel_size * kernel_size)
    kernel = tf.ones([kernel_size, kernel_size], tf.float32)
    kernel = tf.reshape(kernel, (kernel.get_shape().as_list() + [1, 1]))
    kernel = tf.tile(kernel, [1, 1, tf.shape(x)[-1], 1]) * weight
    y = tf.compat.v1.nn.dilation2d(padding_tensor(x, kernel_size),
                                   filter=tf.squeeze(kernel, -1) - weight,
                                   strides=[1, strides, strides, 1], padding="VALID",
                                   rates=[1, dilations, dilations, 1])
    return y


def erosion2d(x, kernel_size=3, dilations=1, strides=1):
    weight = 1. / (kernel_size * kernel_size)
    kernel = tf.ones([kernel_size, kernel_size], tf.float32)
    kernel = tf.reshape(kernel, (kernel.get_shape().as_list() + [1, 1]))
    kernel = tf.tile(kernel, [1, 1, tf.shape(x)[-1], 1]) * weight
    y = tf.compat.v1.nn.erosion2d(padding_tensor(x, kernel_size),
                                  kernel=tf.squeeze(kernel, -1) - weight,
                                  strides=[1, strides, strides, 1], padding="VALID",
                                  rates=[1, dilations, dilations, 1])
    return y


def fixed_soft_skeletonize(x, maximum_iterations=10, kernel_size=3, dilations=1):
    for _ in tf.range(0, maximum_iterations):
        min_pool_x = erosion2d(x, kernel_size=kernel_size, dilations=dilations)
        contour = tf.nn.relu(dilation2d(min_pool_x, kernel_size=kernel_size, dilations=dilations) - min_pool_x)
        x = tf.nn.relu(x - contour)
    return x


def soft_skeletonize(x, maximum_iterations=10, kernel_size=3, dilations=1, threshold=1.):
    _, h, w, c = x.get_shape().as_list()

    def body(_, skelitonize):
        eroded = erosion2d(skelitonize, kernel_size=kernel_size, dilations=dilations)
        skelitonize = tf.nn.relu(
            skelitonize - tf.nn.relu(dilation2d(eroded, kernel_size=kernel_size, dilations=dilations) - eroded))
        skelitonize.set_shape([None, h, w, c])
        eroded.set_shape([None, h, w, c])
        return [eroded, skelitonize]

    def cond(prev_eroded, _):
        return tf.reduce_any(tf.reduce_sum(prev_eroded, (1, 2, 3)) > threshold)

    shape_invariants = [x.get_shape(), tf.TensorShape([None, h, w, c])]
    _, x = tf.compat.v1.while_loop(cond=cond,
                                   body=body,
                                   shape_invariants=shape_invariants,
                                   maximum_iterations=maximum_iterations,
                                   loop_vars=[x, x])
    return x

def soft_cldice_losses(y_true, y_pred, true_skeleton=None, maximum_iterations=10, fixed_iterations=False):
    if fixed_iterations:
        soft_skeletonize_func = fixed_soft_skeletonize
    else:
        soft_skeletonize_func = soft_skeletonize
    pred_skeleton = soft_skeletonize_func(y_pred, maximum_iterations=maximum_iterations)
    if true_skeleton is None:
        true_skeleton = soft_skeletonize_func(y_true, maximum_iterations=maximum_iterations)
    iflat = norm_intersection(pred_skeleton, y_true)
    tflat = norm_intersection(true_skeleton, y_pred)
    loss = 1. - tf.compat.v1.div_no_nan(2. * iflat * tflat, iflat + tflat)
    return loss


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

def clDice_Dice(y_true, y_pred):
    alpha = 0.5
    data_format="channels_last"
    return (alpha * dice_loss(data_format=data_format)(y_true, y_pred) + 
            (1-alpha) * soft_cldice_losses(y_true, y_pred))