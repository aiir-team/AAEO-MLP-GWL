# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import keras.backend as K


def MSE(y_true, y_pred):
    return K.mean(K.pow(y_true - y_pred, 2))

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))

def MAE(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def ME(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))


def round_function(value, decimal=None):
    if decimal is None:
        if 0 < value < 1:
            return round(value, 6)
        else:
            return round(value, 3)
    else:
        return round(value, decimal)