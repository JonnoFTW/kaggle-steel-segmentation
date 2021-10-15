"""Metrics for measuring machine learning algorithm performances
"""

import keras.backend as K
import tensorflow as tf
import numpy as np


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        # y_pred_ = tf.to_int32(y_pred > t)
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def iou(actual, predicted):
    """Compute Intersection over Union statistic (i.e. Jaccard Index)
    See https://en.wikipedia.org/wiki/Jaccard_index
    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels
    Returns
    -------
    float
        Intersection over Union value
    """
    actual = K.flatten(actual)
    predicted = K.flatten(predicted)
    intersection = K.sum(actual * predicted)
    union = K.sum(actual) + K.sum(predicted) - intersection
    print(intersection, type(intersection))
    print(union, type(union))
    return 1. * intersection / union


def iou_loss(actual, predicted):
    """Loss function based on the Intersection over Union (IoU) statistic
    IoU is comprised between 0 and 1, as a consequence the function is set as
    `f(.)=1-IoU(.)`: the loss has to be minimized, and is comprised between 0
    and 1 too
    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels
    Returns
    -------
    float
        Intersection-over-Union-based loss
    """
    return 1. - iou(actual, predicted)


def dice_coef(actual, predicted, eps=1e-3):
    """Dice coef
    See https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Examples at:
      -
    https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L23
      -
    https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/blob/master/zf_unet_224_model.py#L36
    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels
    eps : float
        Epsilon value to add numerical stability
    Returns
    -------
    float
        Dice coef value
    """
    y_true_f = K.flatten(actual)
    y_pred_f = K.flatten(predicted)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (K.sum(y_true_f) + K.sum(y_pred_f) + eps)


def dice_coef_loss(actual, predicted):
    """
    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels
    Returns
    -------
    float
        Dice-coef-based loss
    """
    return -dice_coef(actual, predicted)


def tversky_loss(alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tversky loss function.
    Alpha and beta should be the class disparity in the dataset (hope it's the same for the test set)
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """

    def func(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        truepos = K.sum(y_true * y_pred)
        fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
        answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
        return -answer

    return func

# def one_hot2dist(seg: np.ndarray) -> np.ndarray:
#     int = len(seg)
#     res = np.zeros_like(seg)
#     for c in range(C):
#         posmask = seg[c].astype(np.bool)
#         if posmask.any():
#             negmask = ~posmask
#             res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
#     return res
# def surface_loss(y_true, y_pred):
#
#     pc = probs[:, self.idc, ...]
#     dc = dist_maps[:, self.idc, ...]
#
#     multipled = tf.einsum("bcwh,bcwh->bcwh", pc, dc)
#     divided = tf.einsum('')
#     loss = K.mean(divided)
#     return loss

def tversky_loss_alt(y_true, y_pred, alpha=0.8, beta=0.7):
    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def jaccard_coef_logloss(y_true, y_pred, smooth=1e-10):
    """ Loss function based on jaccard coefficient.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return -K.log(jaccard + smooth)
