import numpy as np
from src.models import LinearRegression
from src.preprocessing import denormalize

def train_val_split(features, target, train_size=0.8, seed=42):
    """
    Divide el conjunto de datos en entrenamiento y validación mediante remuestreo aleatorio.

    Parámetros:
    -----------
    features : np.ndarray
        Matriz de características (X).
    target : np.ndarray
        Vector de la variable objetivo (y).
    train_size : float, opcional
        Proporción de datos para entrenamiento (0 a 1). Por defecto es 0.8.
    seed : int, opcional
        Semilla para asegurar la reproducibilidad del proceso. Por defecto es 42.

    Retorna:
    --------
    tuple : (x_train, x_val, y_train, y_val)
        Los cuatro arreglos resultantes de la división.
    """
    np.random.seed(seed) # Establecer semilla para reproducibilidad
    
    total_samples = features.shape[0]
    # Asegura que todos los registros (desde 0 a n-1) sean considerados.
    indices = np.arange(total_samples)
    np.random.shuffle(indices) # Mezclar índices de forma aleatoria para romper cualquier tipo de orden

    split_limit = int(total_samples * train_size)
    train_indices = indices[:split_limit]
    val_indices = indices[split_limit:]

    x_train = features[train_indices]
    x_val = features[val_indices]
    y_train = target[train_indices]
    y_val = target[val_indices]

    return x_train, x_val, y_train, y_val


def stratified_train_val_split(features, target, strata, train_size=0.8, seed=42):
    """
    Divide el conjunto de datos en entrenamiento y validación de forma estratificada:
    el muestreo aleatorio se realiza por separado dentro de cada estrato, de modo que
    ambos conjuntos preservan por construcción la proporción de cada estrato.

    Parámetros:
    -----------
    features : np.ndarray
        Matriz de características (X).
    target : np.ndarray
        Vector de la variable objetivo (y).
    strata : np.ndarray
        Vector de etiquetas de estrato (una por observación). Por ejemplo, la
        combinación de país y cuartil de precio.
    train_size : float, opcional
        Proporción de datos para entrenamiento (0 a 1). Por defecto es 0.8.
    seed : int, opcional
        Semilla para asegurar la reproducibilidad del proceso. Por defecto es 42.

    Retorna:
    --------
    tuple : (x_train, x_val, y_train, y_val)
        Los cuatro arreglos resultantes de la división.
    """
    np.random.seed(seed)
    strata = np.asarray(strata)

    train_indices, val_indices = [], []

    for stratum in np.unique(strata):
        idx = np.where(strata == stratum)[0]
        np.random.shuffle(idx)

        split_limit = int(round(len(idx) * train_size))
        train_indices.extend(idx[:split_limit])
        val_indices.extend(idx[split_limit:])

    # Mezcla final para que las filas no queden ordenadas por estrato
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    x_train = features[train_indices]
    x_val = features[val_indices]
    y_train = target[train_indices]
    y_val = target[val_indices]

    return x_train, x_val, y_train, y_val


def cross_val(X_norm, y_norm, mu_y, sigma_y, k=5, l1=0, l2=0, lr=0.01, epochs=5000, method='pinv'):
    """
    Realiza una validación cruzada de K-folds para evaluar el desempeño del modelo.
    
    Parámetros de entrada:
    ----------------------
    X_norm : np.ndarray
        Matriz de características normalizada.
    y_norm : np.ndarray
        Vector objetivo normalizado.
    mu_y : float
        Media del vector objetivo original (para desnormalización).
    sigma_y : float
        Desviación estándar del vector objetivo original (para desnormalización).
    k : int, opcional
        Número de pliegues o particiones. Por defecto es 5.
    l1 : float, opcional
        Coeficiente de regularización L1 (Lasso). Por defecto es 0.
    l2 : float, opcional
        Coeficiente de regularización L2 (Ridge). Por defecto es 0.
    lr : float, opcional
        Tasa de aprendizaje para el descenso por gradiente. Por defecto es 0.01.
    epochs : int, opcional
        Número de iteraciones para el entrenamiento por gradiente. Por defecto es 5000.
    method : str, opcional
        Algoritmo de optimización ('pinv' o 'gd'). Por defecto es 'pinv'.

    Parámetros de salida:
    ---------------------
    mean_rmse_usd : float
        Promedio del error RMSE calculado en dólares reales entre todos los pliegues.
    """
    
    indices = np.arange(X_norm.shape[0])
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    rmse_usd_scores = []

    for i in range(k):
        # Selección de índices para el fold actual de validación y entrenamiento
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        # Segmentación de los datos para el fold actual
        X_train_f, X_val_f = X_norm[train_idx], X_norm[val_idx]
        y_train_f, y_val_f = y_norm[train_idx], y_norm[val_idx]
        
        model = LinearRegression(X_train_f, y_train_f, l1=l1, l2=l2)
        
        if method == 'pinv':
            model.fit_pseudo_inverse()
        else:
            model.fit_gradient_descent(lr=lr, epochs=epochs)
            
        y_pred_norm = model.predict(X_val_f)
        
        y_pred_usd = denormalize(y_pred_norm, mu_y, sigma_y)
        y_val_usd = denormalize(y_val_f, mu_y, sigma_y)
        
        rmse_fold = np.sqrt(np.mean((y_val_usd - y_pred_usd)**2))
        rmse_usd_scores.append(rmse_fold)
        
    mean_rmse_usd = np.mean(rmse_usd_scores)
        
    return mean_rmse_usd