import numpy as np

class LinearRegression():
    """    
    Esta clase implementa el modelo de regresión lineal multivariada permitiendo
    la optimización de parámetros mediante el método analítico (Pseudo-inversa)
    y el método numérico (Descenso por Gradiente). Admite regularización L1 y L2.
    
    Atributos:
    ----------
    X : np.ndarray
        Matriz de características aumentada (incluye columna de unos para el bias).
    y : np.ndarray
        Vector objetivo (precios) con forma (n_samples, 1).
    coef : np.ndarray
        Vector de pesos/coeficientes resultantes del entrenamiento.
    l1 : float
        Coeficiente de penalización para regularización Lasso.
    l2 : float
        Coeficiente de penalización para regularización Ridge.
    """

    def __init__(self, X, y, l1=0, l2=0):
        """
        Inicializa la clase y prepara los datos aumentando la matriz X con el bias.
        
        Parámetros de entrada:
        ----------------------
        X : np.ndarray
            Matriz de características de entrada de forma (n_samples, n_features).
        y : np.ndarray
            Vector de etiquetas de forma (n_samples,).
        l1 : float, opcional
            Factor de regularización L1. Por defecto es 0.
        l2 : float, opcional
            Factor de regularización L2. Por defecto es 0.
        """
        ones = np.ones((len(X), 1))
        self.X = np.hstack((ones, X))
        
        # Aseguramos que el target tenga la forma (n, 1) para operaciones matriciales
        self.y = y.reshape(-1, 1)
        self.coef = None
        self.l1 = l1
        self.l2 = l2
        pass

    def fit_pseudo_inverse(self):
        """
        Entrena el modelo utilizando la ecuación normal con la pseudo-inversa. 
        Soporta regularización L2 (Ridge).
        
        Parámetros de salida:
        ---------------------
        None : Actualiza el atributo self.coef con los pesos óptimos calculados.
        """
        n_features = self.X.shape[1]
        
        I = np.eye(n_features)
        I[0, 0] = 0
        
        if self.l2 != 0:
            # Resolución para ridge
            xtx_regularization_term = self.X.T @ self.X + self.l2 * I
            self.coef = np.linalg.pinv(xtx_regularization_term) @ self.X.T @ self.y
        else:
            # Resolución analítica estándar por cuadrados mínimos
            self.coef = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y

    def fit_gradient_descent(self, lr=0.01, epochs=5000):
        """
        Entrena el modelo utilizando el algoritmo de Descenso por Gradiente, 
        incorporando regularización L2 y un mecanismo para reducir a cero los coeficientes menos relevantes en el caso de L1.

        Parámetros de entrada:
        ----------------------
        lr : float, opcional
            Tasa de aprendizaje (learning rate). Por defecto es 0.01.
        epochs : int, opcional
            Número de iteraciones sobre el set de entrenamiento. Por defecto es 5000.
            
        Parámetros de salida:
        ---------------------
        None : Actualiza el atributo self.coef con los pesos resultantes.
        """
        n_samples, n_features = self.X.shape
        self.coef = np.zeros((n_features, 1))

        for epoch in range(epochs):
            predictions = self.X @ self.coef
            errors = predictions - self.y
            
            gradient = (2 / n_samples) * (self.X.T @ errors)
            
            if self.l2 > 0:
                gradient += 2 * self.l2 * self.coef
                gradient[0] = 0 # El bias no se debe penalizar
                
            self.coef = self.coef - lr * gradient
            
            if self.l1 > 0:
                # El umbral determina qué tan agresiva es la reducción de los pesos
                threshold = lr * self.l1
                
                # Se opera solo sobre los coeficientes de las variables, protegiendo el bias (coef[0])
                w = self.coef[1:] 
                
                # La logíca planteada es que restamos el umbral al valor absoluto del peso. 
                # Si el resultado es menor o igual a cero, el peso se anula exactamente (se hace 0.0).
                # Esto permite que el modelo descarte automáticamente las variables menos relevantes.
                self.coef[1:] = np.sign(w) * np.maximum(0, np.abs(w) - threshold)

    def predict(self, X_new):
        """
        Realiza la predicción del target para nuevas observaciones.
        
        Parámetros de entrada:
        ----------------------
        X_new : np.ndarray
            Matriz de nuevas observaciones de forma (n_samples, n_features).
            
        Parámetros de salida:
        ---------------------
        predictions : np.ndarray
            Valores predichos por el modelo entrenado.
        """
        # Aumentar matriz de entrada con columna de unos para el producto escalar con el bias
        X_new_bias = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
        
        return X_new_bias @ self.coef

    def print_coefficients(self, feature_names):
        """
        Muestra por consola los pesos finales asignados a cada atributo.
        
        Parámetros de entrada:
        ----------------------
        feature_names : list
            Lista con los nombres de las variables para facilitar la interpretación.
        """
        if self.coef is None:
            print("Error: El modelo debe ser entrenado antes de mostrar los coeficientes.")
            return

        print("Variable | Weight")
        print(f"Intercepto: {self.coef[0][0]:4f}")

        for name, weight in zip(feature_names, self.coef[1:]):
            print(f"{name}: {weight[0]:4f}")