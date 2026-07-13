import pandas as pd
from src.models import LinearRegression
from src.preprocessing import normalize, denormalize
from src.metrics import rmse
import numpy as np

def get_learning_curve(df_train, df_val, features, label='precio', l1=0, l2=0, method='gd'):
    """
    Genera los datos necesarios para graficar la curva de aprendizaje del modelo,
    evaluando el error en función del tamaño del conjunto de entrenamiento.

    Parámetros de entrada:
    ----------------------
    df_train : pd.DataFrame
        Dataset de entrenamiento completo.
    df_val : pd.DataFrame
        Dataset de validación para evaluar la generalización.
    features : list
        Lista de nombres de las columnas a utilizar como predictores.
    label : str, opcional
        Nombre de la columna objetivo. Por defecto es 'precio'.
    l1 : float, opcional
        Parámetro de regularización Lasso. Por defecto es 0.
    l2 : float, opcional
        Parámetro de regularización Ridge. Por defecto es 0.
    method : str, opcional
        Algoritmo de optimización ('gd' para gradiente, 'pinv' para analítico).

    Parámetros de salida:
    ---------------------
    tuple : (m_sizes, train_errors, val_errors)
        m_sizes: Tamaños de muestra utilizados (eje X).
        train_errors: Historial de RMSE en entrenamiento (eje Y).
        val_errors: Historial de RMSE en validación (eje Y).
    """
    train_errors, val_errors = [], []
    
    X_train_raw = df_train[features].values
    y_train_raw = df_train[label].values.reshape(-1, 1)
    X_val_raw = df_val[features].values
    y_val_raw = df_val[label].values.reshape(-1, 1)
    
    # Definición de los pasos de tamaño de muestra (del 10% al 100% del dataset)
    m_sizes = np.linspace(0.1, 1.0, 10) * len(df_train)
    m_sizes = m_sizes.astype(int)

    for m in m_sizes:
        # Selección del subconjunto de entrenamiento de tamaño m
        X_sub_raw = X_train_raw[:m]
        y_sub_raw = y_train_raw[:m]
        
        X_sub_norm, mu_X, sigma_X = normalize(X_sub_raw)
        y_sub_norm, mu_y, sigma_y = normalize(y_sub_raw)
        
        X_val_norm, _, _ = normalize(X_val_raw, mu=mu_X, sigma=sigma_X)
        
        model = LinearRegression(X_sub_norm, y_sub_norm, l1=l1, l2=l2)
        if method == 'pinv':
            model.fit_pseudo_inverse()
        else:
            model.fit_gradient_descent(lr=0.01, epochs=3000)
            
        p_train_usd = denormalize(model.predict(X_sub_norm), mu_y, sigma_y)
        p_val_usd = denormalize(model.predict(X_val_norm), mu_y, sigma_y)
        
        train_errors.append(rmse(y_sub_raw, p_train_usd))
        val_errors.append(rmse(y_val_raw, p_val_usd))
        
    return m_sizes, train_errors, val_errors



def train_segmented_model(df_train, df_val, features, label='precio', l1=0, l2=0):
    """
    Entrena un modelo de regresión para un subconjunto geográfico específico.

    Parámetros de entrada:
    ----------------------
    df_train : pd.DataFrame
        Dataset de entrenamiento segmentado.
    df_val : pd.DataFrame
        Dataset de validación segmentado.
    features : list
        Lista de las 50 características seleccionadas para el modelo.
    label : str, opcional
        Variable objetivo a predecir. Por defecto es 'precio'.
    l1 : float, opcional
        Coeficiente de regularización L1 (Lasso). Por defecto es 0.
    l2 : float, opcional
        Coeficiente de regularización L2 (Ridge). Por defecto es 0.

    Parámetros de salida:
    ---------------------
    tuple : Contiene el modelo entrenado, parámetros de normalización (mu, sigma)
            y los vectores de valores reales y predichos para Train y Val.
    """
    X_train_raw = df_train[features].values.astype(np.float64)
    y_train_raw = df_train[label].values.reshape(-1, 1)
    X_val_raw = df_val[features].values.astype(np.float64)
    y_val_raw = df_val[label].values.reshape(-1, 1)

    # Normalización local para manejar escalas de precios distintas entre países
    X_train_norm, mu_X, sigma_X = normalize(X_train_raw)
    y_train_norm, mu_y, sigma_y = normalize(y_train_raw)
    
    # Se normaliza validación usando lo obtenidos en entrenamiento (evita Leakage)
    X_val_norm, _, _ = normalize(X_val_raw, mu=mu_X, sigma=sigma_X)

    model = LinearRegression(X_train_norm, y_train_norm, l1=l1, l2=l2)
    model.fit_gradient_descent(lr=0.01, epochs=3000)

    y_pred_train_norm = model.predict(X_train_norm)
    y_pred_val_norm = model.predict(X_val_norm)
    
    y_pred_train_usd = denormalize(y_pred_train_norm, mu_y, sigma_y)
    y_pred_val_usd = denormalize(y_pred_val_norm, mu_y, sigma_y)

    return model, mu_X, sigma_X, mu_y, sigma_y, y_train_raw, y_pred_train_usd, y_val_raw, y_pred_val_usd

def get_learning_curve_segmented(df_train, df_val, features, label='precio'):
    """
    Calcula la curva de aprendizaje para un modelo segmentado geográficamente (USA y ARG).
    
    Esta función itera sobre diferentes tamaños de muestra del conjunto de entrenamiento
    total, entrena modelos independientes para cada región (USA y ARG) y consolida los 
    resultados para obtener métricas globales de error.
    
    Parámetros de entrada:
    ----------------------
    df_train : pd.DataFrame
        Dataset de entrenamiento que debe contener la columna 'is_usa'.
    df_val : pd.DataFrame
        Dataset de validación que debe contener la columna 'is_usa'.
    features : list
        Lista de nombres de las columnas a utilizar como características.
    label : str, opcional
        Nombre de la variable objetivo. Por defecto es 'precio'.
        
    Parámetros de salida:
    ---------------------
    m_sizes : np.ndarray
        Array con los tamaños de muestra (m) utilizados en cada iteración.
    train_errors : list
        Lista con el RMSE global de entrenamiento para cada tamaño m.
    val_errors : list
        Lista con el RMSE global de validación para cada tamaño m.
    """
    train_errors, val_errors = [], []
    
    # Tamaños de entrenamiento (del 10% al 100%)
    m_sizes = (np.linspace(0.1, 1.0, 10) * len(df_train)).astype(int)

    for m in m_sizes:
        # Tomamos el subconjunto de entrenamiento total
        df_sub = df_train.iloc[:m]
        
        train_usa = df_sub[df_sub['is_usa'] == 1]
        train_arg = df_sub[df_sub['is_usa'] == 0]
        val_usa = df_val[df_val['is_usa'] == 1]
        val_arg = df_val[df_val['is_usa'] == 0]

        # Entrenamos y predecimos para cada segmento
        res_usa = train_segmented_model(train_usa, val_usa, features) if len(train_usa) > 0 else None
        res_arg = train_segmented_model(train_arg, val_arg, features) if len(train_arg) > 0 else None

        # Consolidamos predicciones para el RMSE Global
        # Si un segmento no tiene datos en el paso 'm', se ignora para ese cálculo
        y_t_raw = []
        y_t_pred = []
        y_v_raw = []
        y_v_pred = []

        for res in [res_usa, res_arg]:
            if res:
                y_t_raw.append(res[5])
                y_t_pred.append(res[6])
                y_v_raw.append(res[7])
                y_v_pred.append(res[8])

        train_errors.append(rmse(np.vstack(y_t_raw), np.vstack(y_t_pred)))
        val_errors.append(rmse(np.vstack(y_v_raw), np.vstack(y_v_pred)))

    return m_sizes, train_errors, val_errors

def predict_segment(df, model, features, mu_X, sigma_X, mu_y, sigma_y):
    """
    Aplica normalización, predicción y desnormalización para un segmento geográfico.
    
    Esta función procesa un subconjunto de datos utilizando los parámetros de 
    normalización (media y desvío) obtenidos durante la etapa de entrenamiento 
    específica de ese segmento.

    Parámetros de entrada:
    ----------------------
    df : pd.DataFrame
        Dataset del segmento a evaluar (ej. solo USA o solo ARG).
    model : LinearRegression
        Instancia del modelo entrenado para este segmento específico.
    features : list
        Lista de nombres de las características (features) utilizadas.
    mu_X : np.ndarray
        Media de las características obtenida en el entrenamiento.
    sigma_X : np.ndarray
        Desvío estándar de las características obtenido en el entrenamiento.
    mu_y : float
        Media de la variable objetivo obtenida en el entrenamiento.
    sigma_y : float
        Desvío estándar de la variable objetivo obtenido en el entrenamiento.

    Parámetros de salida:
    ---------------------
    tuple : (y_raw, y_pred_usd)
        y_raw : np.ndarray
            Valores reales del segmento en escala original.
        y_pred_usd : np.ndarray
            Valores predichos por el modelo en escala original (USD).
    """
    X_raw = df[features].values.astype(np.float64)
    y_raw = df['precio'].values.reshape(-1, 1)
    
    # Normalización con parámetros de train del segmento correspondiente
    X_norm, _, _ = normalize(X_raw, mu=mu_X, sigma=sigma_X)
    
    y_pred_norm = model.predict(X_norm)
    y_pred_usd = denormalize(y_pred_norm, mu_y, sigma_y)
    
    return y_raw, y_pred_usd