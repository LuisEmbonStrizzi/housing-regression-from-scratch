# TP1: Regresión Lineal - Aprendizaje Automático y Aprendizaje Profundo

Este proyecto tiene como objetivo desarrollar y evaluar diversos modelos de regresión lineal para estimar el precio de venta de viviendas, implementando los algoritmos desde cero utilizando únicamente **numpy**

## 📊 Descripción del Proyecto
El trabajo aborda el ciclo completo de un proyecto de regresión lineal:
1. **Análisis Exploratorio de Datos (EDA):** Limpieza, visualización y análisis de correlaciones.
2. **Implementación de Modelos:** Creación de una clase de Regresión Lineal con métodos de Pseudo-inversa y Descenso por Gradiente.
3. **Feature Engineering:** Construcción de características derivadas para mejorar la capacidad predictiva.
4. **Regularización:** Implementación de penalizaciones L1 (Lasso) y L2 (Ridge) y aplicación a modelos.
5. **Evaluación:** Selección del mejor modelo mediante learning curves y validación cruzada.


## 📂 Estructura de Archivos
Se sigue la estructura modular sugerida para facilitar la corrección y reutilización del código:

```text
udesa-ml-tp1-regresion-lineal/
├── data/
│   ├── raw/                        # Datasets originales (casas_dev.csv, casas_test.csv)
│   └── processed/                  # Datos generados por el notebook (no versionados)
├── src/                            # Código fuente modularizado
│   ├── __init__.py                 # Define la carpeta como paquete Python
│   ├── models.py                   # Clase LinearRegression (Gradiente y Pseudo-inversa)
│   ├── metrics.py                  # Funciones de error: MSE, MAE, RMSE, R2
│   ├── preprocessing.py            # Normalización, limpieza e imputación (estadísticos de train)
│   ├── feature_engineering.py      # Manejo del flujo de feature engineering
│   ├── utils.py                    # Funciones útiles para graficar y entrenar modelos
│   └── data_splitting.py           # Split aleatorio y estratificado, cross-validation
├── notebooks/
│   └── Entrega_TP1.ipynb           # Notebook con respuestas y gráficos finales
├── requirements.txt                # Especificación de dependencias
└── README.md                       # Descripción e instrucciones
```

## ⚙️ Instalación y Uso

### 1. Configuración del Entorno
Primero, usamos un entorno virtual para evitar conflictos de dependencias y luego ejecutamos:
```bash
pip install -r requirements.txt
```

### 2. Ejecución
Para visualizar los resultados y las respuestas a los ejercicios:
1. Navegar a la carpeta `notebooks/`.
2. Abrir `Entrega_TP1.ipynb` en Jupyter o VS Code.
3. Configurar el kernel de python y correrlo.

El notebook ya incluye todos los outputs guardados, por lo que puede leerse sin ejecutarlo. Si se ejecuta, los archivos de `data/processed/` se regeneran automáticamente.

## 📝 Autor
* **Estudiante:** Luis Augusto Embon Strizzi
* **Materia:** I302 - Aprendizaje Automático y Aprendizaje Profundo
* **Fecha de entrega:** 27 de marzo de 2026

