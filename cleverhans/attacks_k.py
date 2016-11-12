import copy
import keras
import math
import numpy as np
from keras import backend as K

from utils import batch_indices, k_model_loss

import gflags
FLAGS = gflags.FLAGS

def fgsm(x, predictions, eps, clip_min=None, clip_max=None):
    """
    TensorFlow implementation of the Fast Gradient 
    Sign method. 
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """
    K.cast(K.equal(predictions, K.max(predictions, 1, keep_dims=True)),'float32')
    y=y/K.sum(y,1,keep_dims=True)
    loss = k_model_loss(y, predictions, mean=False)
    
    # Define gradient of loss wrt input
    grad, = K.gradients(loss, x)

    signed_grad = K.sign(grad)

    scaled_signed_grad = eps * signed_grad

    adv_x = K.stop_gradient(x + scaled_signed_grad)

    if (clip_min is not None) and (clip_max is not None):
        adv_x = K.clip(adv_x, clip_min, clip_max)
