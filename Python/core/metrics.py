from functools import partial

from keras import backend as K
import tensorflow as tf
import numpy as np
from unet.unet_config import get_kfold_configuration, get_unet_configuration


def update_global_weights(mconfig):
    # Added global variables here for increased efficiency
    global dice_weights
    global ce_weights
    global n_labels

    n_labels = mconfig["n_labels"]
    dice_weights = tf.reshape([w / sum(mconfig["label_weights"]) for w in mconfig["label_weights"]],
                              shape=[n_labels, 1])  # normalized weigths
    ce_weights = tf.reshape(dice_weights,
                            shape=(n_labels,) + tuple(np.ones(len(mconfig['image_shape'])).astype(int)))


def soft_dice_coeff(y_true, y_pred, summation_axis=(-3, -2, -1), epsilon=1e-7):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.

    # Arguments
        y_true:
        y_pred:
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge
        https://arxiv.org/abs/1802.10508
        An overview of semantic image segmentation
        https://www.jeremyjordan.me/semantic-segmentation/
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
    '''

    # find numerator and denominator
    numerator = 2. * K.sum(y_true * y_pred, axis=summation_axis) + epsilon
    denominator = K.sum(y_true, axis=summation_axis) + K.sum(y_pred,
                                                             axis=summation_axis) + epsilon  # y_true and y_pred can also be squared with K.square()
    # calulate dice for each class
    dice = numerator / denominator
    return dice


def weighted_soft_dice_coeff(y_true, y_pred):
    dice = soft_dice_coeff(y_true, y_pred)
    dice = tf.reshape(dice, [1, n_labels])  # TODO: Make able to handle batches larger than 1
    weighted_dice = dice @ dice_weights
    return tf.reshape(weighted_dice, ())


def weighted_soft_dice_loss(y_true, y_pred):
    return 1 - weighted_soft_dice_coeff(y_true, y_pred)


def generalized_dice_coeff(y_true, y_pred, summation_axis=(-3, -2, -1), epsilon=1e-7):
    """ Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    :param y_true:
    :param y_pred:
    :param summation_axis:
    :param epsilon:
    :return:
    """
    # GDL weighting: the contribution of each label is corrected by the inverse of its volume
    counts = K.sum(y_true, axis=summation_axis)  # weight for labels
    w_l = 1 / (counts * counts)
    w_l = tf.where(tf.is_finite(w_l), w_l, tf.ones_like(w_l) * epsilon)  # if count=0; weight=epsilon

    numerator = 2. * K.sum(y_true * y_pred, axis=summation_axis)
    numerator = numerator * w_l

    denominator = K.sum(y_true, axis=summation_axis) + K.sum(y_pred, axis=summation_axis)
    denominator = K.clip(denominator * w_l, max_value=None, min_value=epsilon)

    return K.sum(numerator) / K.sum(denominator)


def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)


def cross_entropy_loss(y_true, y_pred, class_axis=1, epsilon=1e-7):
    """
    Categorical crossentropy loss
    :param y_true:
    :param y_pred:
    :param class_axis:
    :param epsilon:
    :return:
    """
    # scale predictions so that the class probabilities of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=class_axis, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    # calculate entropy
    pixel_and_class_loss = y_true * K.log(y_pred) * ce_weights  # a value for every class in each pixel
    pixel_loss = -K.sum(pixel_and_class_loss, axis=class_axis)  # cross entropy for each pixel
    # reduce by averaging the entropy
    loss = K.mean(pixel_loss)
    return loss


def numpy_soft_dice_coeff(y_true, y_pred, axis=(-3, -2, -1), epsilon=1e-6):
    # find numerator and denominator
    numerator = 2. * np.sum(y_true * y_pred, axis=axis) + epsilon
    denominator = np.sum(y_true, axis=axis) + np.sum(y_pred,
                                                     axis=axis) + epsilon  # y_true and y_pred can also be squared with K.square()
    # calulate dice for each class
    dice = numerator / denominator
    return dice
