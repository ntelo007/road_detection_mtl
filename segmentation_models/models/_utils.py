<<<<<<< HEAD
from keras_applications import get_submodules_from_kwargs


def freeze_model(model, **kwargs):
    """Set all layers non trainable, excluding BatchNormalization layers"""
    _, layers, _, _ = get_submodules_from_kwargs(kwargs)
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return
=======
from keras_applications import get_submodules_from_kwargs


def freeze_model(model, **kwargs):
    """Set all layers non trainable, excluding BatchNormalization layers"""
    _, layers, _, _ = get_submodules_from_kwargs(kwargs)
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return
>>>>>>> 3ae8ec08729e4099672edda4ac7106879b1b8c92
