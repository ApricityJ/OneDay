import numpy as np

from sklearn.metrics import f1_score, precision_recall_curve
from scipy.misc import derivative


# def sigmoid(x): return 1. / (1. + np.exp(-x))
#
#
# def softmax(x):
#     exp_x = np.exp(x - np.max(x))
#     return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-6)


def best_threshold(y_true, pred_proba, proba_range, verbose=False):
    """
    Function to find the probability threshold that optimises the f1_score

    Comment: this function is not used in this repo, but I include it in case the
    it useful

    Parameters:
    -----------
    y_true: numpy.ndarray
        array with the true labels
    pred_proba: numpy.ndarray
        array with the predicted probability
    proba_range: numpy.ndarray
        range of probabilities to explore.
        e.g. np.arange(0.1,0.9,0.01)

    Return:
    -----------
    tuple with the optimal threshold and the corresponding f1_score
    """
    scores = []
    for prob in proba_range:
        pred = [int(p > prob) for p in pred_proba]
        score = f1_score(y_true, pred)
        scores.append(score)
        if verbose:
            print("INFO: prob threshold: {}.  score :{}".format(round(prob, 3), round(score, 5)))
    best_score = scores[np.argmax(scores)]
    optimal_threshold = proba_range[np.argmax(scores)]
    return (optimal_threshold, best_score)


# def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
#     """
#     Focal Loss for lightgbm
#
#     Parameters:
#     -----------
#     y_pred: numpy.ndarray
#         array with the predictions
#     dtrain: lightgbm.Dataset
#     alpha, gamma: float
#         See original paper https://arxiv.org/pdf/1708.02002.pdf
#     """
#     a, g = alpha, gamma
#     y_true = dtrain.label
#
#     def fl(x, t):
#         p = 1 / (1 + np.exp(-x))
#         return -(a * t + (1 - a) * (1 - t)) * ((1 - (t * p + (1 - t) * (1 - p))) ** g) * (
#                     t * np.log(p) + (1 - t) * np.log(1 - p))
#
#     partial_fl = lambda x: fl(x, y_true)
#     grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
#     hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
#     return grad, hess


# def f1_loss(y_pred, dtrain):
#
#     y = dtrain.label
#     pred = y_pred
#
#     beta = 2
#     p = 1. / (1 + np.exp(-pred))
#     grad = p * ((beta - 1) * y + 1) - beta * y
#     hess = ((beta - 1) * y + 1) * p * (1.0 - p)
#
#     return grad, hess


# def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
#     """
#     Adapation of the Focal Loss for lightgbm to be used as evaluation loss
#
#     Parameters:
#     -----------
#     y_pred: numpy.ndarray
#         array with the predictions
#     dtrain: lightgbm.Dataset
#     alpha, gamma: float
#         See original paper https://arxiv.org/pdf/1708.02002.pdf
#     """
#     a, g = alpha, gamma
#     y_true = dtrain.label
#     p = 1 / (1 + np.exp(-y_pred))
#     loss = -(a * y_true + (1 - a) * (1 - y_true)) * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g) * (
#                 y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
#     return 'focal_loss', np.mean(loss), False


# def lgb_focal_f1_score(preds, lgbDataset):
#     """
#     Adaptation of the implementation of the f1 score to be used as evaluation
#     score for lightgbm. The adaptation is required since when using custom losses
#     the row prediction needs to passed through a sigmoid to represent a
#     probability
#
#     Parameters:
#     -----------
#     preds: numpy.ndarray
#         array with the predictions
#     lgbDataset: lightgbm.Dataset
#     """
#     preds = sigmoid(preds)
#     binary_preds = [int(p > 0.5) for p in preds]
#     y_true = lgbDataset.get_label()
#     return 'f1', f1_score(y_true, binary_preds), True


# def focal_loss_lgb_multi(y_pred, dtrain, alpha, gamma, num_class):
#     a, g = alpha, gamma
#     y_true = dtrain.label
#     # N observations x num_class arrays
#     y_true = np.eye(num_class)[y_true.astype('int')]
#     #print(y_pred.shape)
#     y_pred = y_pred.reshape(-1, num_class, order='F')
#
#     # alpha and gamma multiplicative factors with BCEWithLogitsLoss
#     def fl(x, t):
#         p = 1 / (1 + np.exp(-x))
#         return -(a * t + (1 - a) * (1 - t)) * ((1 - (t * p + (1 - t) * (1 - p))) ** g) * (
#                     t * np.log(p) + (1 - t) * np.log(1 - p))
#
#     partial_fl = lambda x: fl(x, y_true)
#     grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
#     hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
#     # flatten in column-major (Fortran-style) order
#     return grad.flatten('F'), hess.flatten('F')


# def focal_loss_lgb_eval_error_multi(y_pred, dtrain, alpha, gamma, num_class):
#     a, g = alpha, gamma
#     y_true = dtrain.label
#     y_true = np.eye(num_class)[y_true.astype('int')]
#     y_pred = y_pred.reshape(-1, num_class, order='F')
#     p = 1 / (1 + np.exp(-y_pred))
#     loss = -(a * y_true + (1 - a) * (1 - y_true)) * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g) * (
#                 y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
#     # a variant can be np.sum(loss)/num_class
#     return 'focal_loss', np.mean(loss), False


# def lgb_focal_f1_score_multi(preds, lgbDataset):
#     # print(preds)
#     # print(preds.shape)
#     preds = preds.reshape(-1, 10, order='F')
#     # print(preds.shape)
#     multi_preds = np.argmax(softmax(preds), axis=1)
#     y_true = lgbDataset.get_label()
#     return 'f1', f1_score(y_true, multi_preds, average='macro'), True


def get_best_f1_threshold(y_pred, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    numerator = 2 * recall * precision
    denominator = recall + precision
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=(denominator != 0))
    return np.max(f1_scores), thresholds[np.argmax(f1_scores)]


def get_best_f2_threshold(y_pred, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    numerator = 5 * recall * precision
    denominator = recall + 4 * precision
    f2_scores = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=(denominator != 0))
    return np.max(f2_scores), thresholds[np.argmax(f2_scores)]


def lgb_f1_score_eval(y_pred, data):
    y_true = data.get_label()
    f1, threshold = get_best_f1_threshold(y_pred, y_true)
    return 'f1', f1, True


def lgb_f2_score_eval(y_pred, data):
    y_true = data.get_label()
    f2, threshold = get_best_f2_threshold(y_pred, y_true)
    return 'f2', f2, True


def lgb_f1_score_multi_macro_eval(y_pred, data, num_class):
    y_pred = y_pred.reshape((-1, num_class), order='F')
    predictions = np.argmax(y_pred, axis=1)
    y_true = data.get_label()
    return 'f1-macro', f1_score(y_true, predictions, average='macro'), True


def lgb_f1_score_multi_weighted_eval(y_pred, data, num_class):
    y_pred = y_pred.reshape((-1, num_class), order='F')
    predictions = np.argmax(y_pred, axis=1)
    y_true = data.get_label()
    return 'f1-weighted', f1_score(y_true, predictions, average='weighted'), True


def xgb_f1_score_eval(y_pred, data):
    y_true = data.get_label()
    f1, threshold = get_best_f1_threshold(y_pred, y_true)
    return 'f1', f1


def xgb_f1_score_multi_macro_eval(y_pred, data, num_class):
    # y_pred = y_pred.reshape((-1, num_class), order='F')
    predictions = np.argmax(y_pred, axis=1)
    y_true = data.get_label()
    return 'f1-macro', f1_score(y_true, predictions, average='macro')


def xgb_f1_score_multi_weighted_eval(y_pred, data, num_class):
    # y_pred = y_pred.reshape((-1, num_class), order='F')
    predictions = np.argmax(y_pred, axis=1)
    y_true = data.get_label()
    return 'f1-weighted', f1_score(y_true, predictions, average='weighted')
