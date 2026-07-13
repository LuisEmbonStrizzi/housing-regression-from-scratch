import numpy as np
import pandas as pd

def apply_feature_engineering(df, medians_dict=None):
    """
    Realiza la transformación de variables y creación de nuevos atributos (interacciones y potencias).
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Conjunto de datos original con las columnas base.
    medians_dict : dict, opcional
        Diccionario con las medianas calculadas en entrenamiento para evitar el target leakage. 
        Si es None, se calculan a partir de los datos de entrada (uso exclusivo en Train).

    Retorna:
    --------
    tuple : (df_fe, medians_dict)
        El DataFrame transformado y el diccionario de medianas utilizado.
    """
    df_fe = df.copy()
    
    columns_to_convert = ['metros_cubiertos', 'lat', 'precio', 'ambientes']
    for col in columns_to_convert:
        if col in df_fe.columns:
            df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce')
    
    df_fe['metros_cubiertos_2'] = df_fe['metros_cubiertos'] ** 2
    df_fe['log_m2'] = np.log1p(df_fe['metros_cubiertos'])
    
    df_fe['is_usa'] = np.where(df_fe['lat'] > 0, 1.0, 0.0)
    df_fe['interact_m2_usa'] = df_fe['metros_cubiertos'] * df_fe['is_usa']
    df_fe['country'] = np.where(df_fe['lat'] > 0, 'usa', 'arg')
    
    if medians_dict is None:
        # Se calculan las medianas solo si no fueron provistas (fase de entrenamiento)
        price_medians = df_fe.groupby('country')['precio'].median().to_dict()
        area_medians = df_fe.groupby('country')['metros_cubiertos'].median().to_dict()
        medians_dict = {'prices': price_medians, 'areas': area_medians}
    
    # Aplicación de las medianas de referencia (evita el data leakage en val y test)
    mapped_price_median = df_fe['country'].map(medians_dict['prices'])
    mapped_area_median = df_fe['country'].map(medians_dict['areas'])
    
    df_fe['coef_zona'] = mapped_price_median / mapped_area_median
    df_fe['precio_estimado_cuadratico'] = (df_fe['metros_cubiertos']**2) * df_fe['coef_zona']

    return df_fe, medians_dict
