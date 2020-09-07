from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

__all__ = ['get_error']


def get_error(error_type, y_true, y_pred):

    error = 0

    if error_type == 'mae':
        # mean absolute error
        error = mean_absolute_error(y_true, y_pred)
            
    elif error_type == 'mse':
        # mean squared error
        error = mean_squared_error(y_true, y_pred)

    elif error_type == 'rmse':
        # root mean square error
        error = sqrt(mean_squared_error(y_true, y_pred))

    else:
        raise ValueError(error_type)

    return error