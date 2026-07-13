import numpy as np
import pandas as pd

def handle_missing_values(df, stats=None):
    """
    Realiza la limpieza de valores nulos mediante eliminación selectiva e imputación.

    Parámetros de entrada:
    ----------------------
    df : pd.DataFrame
        Dataset con posibles valores faltantes.
    stats : dict, opcional
        Estadísticos de imputación calculados sobre el conjunto de entrenamiento
        (claves: 'edad_mediana', 'pisos_moda'). Si es None, se calculan sobre df
        (solo debe ocurrir cuando df es el conjunto de entrenamiento).

    Parámetros de salida:
    ---------------------
    tuple : (df, stats)
        Dataset con nulos gestionados (precio eliminado, edad y pisos imputados)
        y el diccionario de estadísticos utilizados.
    """
    df = df.dropna(subset=['precio']).copy()

    # Consistencia de tipos: registros con ruido o texto no convertible pasan a NaN
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['pisos'] = pd.to_numeric(df['pisos'], errors='coerce')

    # Los estadísticos de imputación se calculan una única vez (sobre train) y se
    # reutilizan como parámetros fijos en validación y test para evitar data leakage
    if stats is None:
        stats = {
            'edad_mediana': df['edad'].median(),
            'pisos_moda': df['pisos'].mode()[0],
        }

    df['edad'] = df['edad'].fillna(stats['edad_mediana'])
    df['pisos'] = df['pisos'].fillna(stats['pisos_moda'])

    return df, stats


def clean_invalid_data(df):
    """
    Filtra registros que presentan inconsistencias económicas o datos imposibles.
    
    Parámetros de entrada:
    ----------------------
    df : pd.DataFrame
        Dataset a filtrar.
        
    Parámetros de salida:
    ---------------------
    df_clean : pd.DataFrame
        Dataset filtrado (solo valores con precio positivo).
    """
    df_clean = df.copy()
    
    # Se eliminan propiedades con precio 0 o negativo por ser errores de carga
    df_clean = df_clean[df_clean['precio'] > 0]
            
    return df_clean


def convert_units(df):
    """
    Estandariza todas las medidas de superficie a metros cuadrados (m2).

    Parámetros de entrada:
    ----------------------
    df : pd.DataFrame
        Dataframe con unidades mixtas (sqft y m2).

    Parámetros de salida:
    ---------------------
    df : pd.DataFrame
        Dataframe con todas las áreas unificadas en la unidad métrica m2.
    """
    # Factor de conversión: 1 sqft = 0.0929 m2
    sqft_to_m2 = 0.0929

    mask = df['unidades'] == 'sqft'

    df.loc[mask, 'Área'] *= sqft_to_m2
    df.loc[mask, 'metros_cubiertos'] *= sqft_to_m2

    # actualización de la etiqueta de unidad
    df.loc[:, 'unidades'] = 'm2'

    return df


def transform_boolean_variables(df):
    """
    Convierte variables categóricas binarias (True/False) a formato numérico (1/0).

    Parámetros de entrada:
    ----------------------
    df : pd.DataFrame
        Dataset con columnas de tipo booleano.

    Parámetros de salida:
    ---------------------
    df : pd.DataFrame
        Dataset con representación entera de variables booleanas.
    """
    if 'pileta' in df.columns:
        df['pileta'] = df['pileta'].astype(int)

    return df


def one_hot_encoder(df, column, categories=None):
    """
    Codifica una variable categórica mediante el método One-Hot Encoding.

    Parámetros de entrada:
    ----------------------
    df : pd.DataFrame
        Dataset original.
    column : str
        Nombre de la columna a transformar.
    categories : list, opcional
        Lista de categorías permitidas (útil para aplicar en Test lo aprendido en Train).

    Parámetros de salida:
    ---------------------
    tuple : (df, categories)
        Dataset con nuevas columnas binarias y la lista de categorías utilizada.
    """
    if categories is None:
        categories = sorted(df[column].unique().tolist())
    
    # Creación de variables Dummy omitiendo la primera para evitar colinealidad
    for cat in categories[1:]:
        new_col_name = f"{column}_{cat}"
        df[new_col_name] = (df[column] == cat).astype(int)
    
    # Removemos la variable categórica original tras la codificación
    df = df.drop(columns=[column])
    
    return df, categories


def normalize(X, mu=None, sigma=None):
    """
    Aplica la normalización Z-score (Estandarización) a la matriz de datos.

    Parámetros de entrada:
    ----------------------
    X : np.ndarray
        Matriz de características a normalizar.
    mu : np.ndarray, opcional
        Vector de medias. Si es None, se calcula sobre X.
    sigma : np.ndarray, opcional
        Vector de desviaciones estándar. Si es None, se calcula sobre X.

    Parámetros de salida:
    ---------------------
    tuple : (X_norm, mu, sigma)
        Matriz normalizada y los parámetros estadísticos utilizados.
    """
    X = np.array(X, dtype=np.float64)
        
    # Cálculo de parámetros si estamos en fase de entrenamiento
    if mu is None or sigma is None:
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        
        sigma = np.where(sigma == 0, 1, sigma)
        
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma


def denormalize(X_norm, mu, sigma):
    """
    Revierte la normalización Z-score para retornar los datos a su escala original.

    Parámetros de entrada:
    ----------------------
    X_norm : np.ndarray
        Datos en escala normalizada.
    mu : np.ndarray
        Media original utilizada en la normalización.
    sigma : np.ndarray
        Desviación estándar original utilizada en la normalización.

    Parámetros de salida:
    ---------------------
    X_orig : np.ndarray
        Datos re-escalados a sus unidades originales (ej: dólares o m2).
    """
    return (X_norm * sigma) + mu

def preprocess_data(df, stats=None):
    """
    Ejecuta el pipeline de limpieza inicial de datos para asegurar la integridad del dataset.

    Este proceso incluye la gestión de valores nulos (imputación de edad y pisos),
    la conversión de unidades métricas y la eliminación de registros con datos
    inconsistentes o erróneos detectados en el EDA.

    Parámetros de entrada:
    ----------------------
    df : pd.DataFrame
        Dataset.
    stats : dict, opcional
        Estadísticos de imputación calculados sobre el conjunto de entrenamiento.
        Debe pasarse siempre que df sea el conjunto de validación o de test, para
        que la imputación no use información de esos conjuntos (data leakage).

    Retorna:
    --------
    tuple : (df, stats)
        Dataset procesado y listo para la fase de transformación de características,
        junto con los estadísticos de imputación utilizados.
    """
    df = df.copy()

    df, stats = handle_missing_values(df, stats=stats)

    df = convert_units(df)

    df = clean_invalid_data(df)

    return df, stats


def refine_features(df, cats_tipo=None):
    """
    Ejecuta el pipeline de transformación final y codificación de variables categóricas.
    
    Se encarga de convertir variables booleanas a formato numérico y aplicar One-Hot Encoding 
    a las categorías de 'tipo' de propiedad.

    Parámetros de entrada:
    ----------------------
    df : pd.DataFrame
        Dataset previamente preprocesado por la función preprocess_data.
    cats_tipo : list, opcional
        Lista de categorías predefinidas para asegurar que el encoding sea consistente 
        entre los conjuntos de entrenamiento y prueba (evitando data leakage).

    Retorna:
    --------
    df : pd.DataFrame
        Dataset con variables numéricas y columnas dummy listas para el entrenamiento.
    categories : list
        Lista de las categorías utilizadas en el One-Hot Encoding para futura referencia.
    """
    df = df.copy()
    
    df = transform_boolean_variables(df)
    
    # Vectorización de la variable 'tipo' (ej: Casa, PH, Departamento)
    # Se utiliza cats_tipo para garantizar que las dimensiones de X coincidan en train y test
    df, categories = one_hot_encoder(df, 'tipo', categories=cats_tipo)
    
    return df, categories