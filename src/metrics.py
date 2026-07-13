import numpy as np

def mse(y_true, y_pred):
    """
    Calcula el Error Cuadrático Medio (MSE) entre los valores reales y las predicciones.
    
    Parámetros de entrada:
    ----------------------
    y_true : np.ndarray
        Vector con los valores reales (target).
    y_pred : np.ndarray
        Vector con los valores predichos por el modelo.
        
    Parámetros de salida:
    ---------------------
    mse_value : float
        Valor del Error Cuadrático Medio resultante.
    """
    error_diff = y_true - y_pred
    mse_value = np.mean(error_diff**2)
    
    return mse_value


def mae(y_true, y_pred):
    """
    Calcula el Error Medio Absoluto (MAE) en unidades originales.
    
    Parámetros de entrada:
    ----------------------
    y_true : np.ndarray
        Vector con los valores reales.
    y_pred : np.ndarray
        Vector con los valores predichos.
        
    Parámetros de salida:
    ---------------------
    mae_value : float
        Promedio de los errores absolutos.
    """
    absolute_diff = np.abs(y_true - y_pred)
    mae_value = np.mean(absolute_diff)
    
    return mae_value


def rmse(y_true, y_pred):
    """
    Calcula la Raíz del Error Cuadrático Medio (RMSE).
    
    Parámetros de entrada:
    ----------------------
    y_true : np.ndarray
        Vector con los valores reales.
    y_pred : np.ndarray
        Vector con los valores predichos.
        
    Parámetros de salida:
    ---------------------
    rmse_value : float
        Raíz cuadrada del MSE calculado.
    """
    mse_val = mse(y_true, y_pred)
    rmse_value = np.sqrt(mse_val)
    
    return rmse_value


def r2(y_true, y_pred):
    """
    Calcula el coeficiente de determinación R^2.
    
    Parámetros de entrada:
    ----------------------
    y_true : np.ndarray
        Vector con los valores reales.
    y_pred : np.ndarray
        Vector con los valores predichos.
        
    Parámetros de salida:
    ---------------------
    r2_value : float
        Proporción de la varianza total explicada por el modelo.
    """
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    
    target_mean = np.mean(y_true)
    total_sum_squares = np.sum((y_true - target_mean) ** 2)
    
    r2_value = 1 - (residual_sum_squares / total_sum_squares)
    
    return r2_value