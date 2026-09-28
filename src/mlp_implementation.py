import numpy as np

# ==========================================
# 1. FUNCIONES DE ACTIVACIÓN Y DERIVADAS
# ==========================================
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1.0, 0.0)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

# ==========================================
# 2. CLASE MLP BASE (1 Capa Oculta)
# ==========================================
class SimpleMLP:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, activation='relu'):
        self.activation_type = activation
        scale = 0.5 if activation == 'sigmoid' else 0.1
        self.W1 = np.random.randn(input_size, hidden_size) * scale
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * scale
        self.b2 = np.zeros((1, output_size))

    def _act(self, z):
        return relu(z) if self.activation_type == 'relu' else sigmoid(z)

    def _act_deriv(self, z):
        return relu_derivative(z) if self.activation_type == 'relu' else sigmoid_derivative(z)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._act(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2
        return self.A2

    def train(self, X, y, learning_rate=0.01, epochs=1000, print_interval=200):
        m = X.shape[0]
        history = []
        for epoch in range(epochs):
            # Forward
            output = self.forward(X)
            loss = np.mean(np.square(output - y))
            history.append(loss)

            # Backward
            error = (output - y) / m
            dW2 = np.dot(self.A1.T, error)
            db2 = np.sum(error, axis=0, keepdims=True)

            d_hidden = np.dot(error, self.W2.T) * self._act_deriv(self.Z1)
            dW1 = np.dot(X.T, d_hidden)
            db1 = np.sum(d_hidden, axis=0, keepdims=True)

            # Actualización
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            if print_interval and epoch % print_interval == 0:
                print(f"  Época {epoch:4d} | Pérdida (Loss): {loss:.6f}")

        final_loss = np.mean(np.square(self.forward(X) - y))
        return history, final_loss

# ==========================================
# 3. MLP DE 4 CAPAS (Actividad 3)
# ==========================================
class DeepMLP4Layers:
    def __init__(self, input_size=2, hidden1_size=4, hidden2_size=4, output_size=1):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.5
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.5
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.5
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.Z3
        return self.A3

    def train(self, X, y, learning_rate=0.05, epochs=2000, print_interval=400):
        m = X.shape[0]
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(np.square(output - y))

            # Backward en cascada
            error3 = (output - y) / m
            dW3 = np.dot(self.A2.T, error3)
            db3 = np.sum(error3, axis=0, keepdims=True)

            error2 = np.dot(error3, self.W3.T) * relu_derivative(self.Z2)
            dW2 = np.dot(self.A1.T, error2)
            db2 = np.sum(error2, axis=0, keepdims=True)

            error1 = np.dot(error2, self.W2.T) * relu_derivative(self.Z1)
            dW1 = np.dot(X.T, error1)
            db1 = np.sum(error1, axis=0, keepdims=True)

            # Actualización
            self.W3 -= learning_rate * dW3
            self.b3 -= learning_rate * db3
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

            if print_interval and epoch % print_interval == 0:
                print(f"  Época {epoch:4d} | Pérdida (Loss): {loss:.6f}")

# ==========================================
# 4. EJECUCIÓN DE PRUEBAS
# ==========================================
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print("=== 1. EXPERIMENTO BASE (ReLU, lr=0.1) ===")
    np.random.seed(42)
    mlp_base = SimpleMLP(2, 4, 1, activation='relu')
    mlp_base.train(X, y, learning_rate=0.1, epochs=1000)
    print("Predicciones finales:")
    print(np.round(mlp_base.forward(X), 4))

    print("\n=== 2. ACTIVIDAD 1: LEARNING RATES (0.9 y 0.0001) ===")
    for lr in [0.9, 0.0001]:
        print(f"\n--- Probando lr = {lr} ---")
        np.random.seed(42)
        m = SimpleMLP(2, 4, 1, activation='relu')
        _, fl = m.train(X, y, learning_rate=lr, epochs=1000, print_interval=400)
        print(f"Loss final: {fl:.6f}")
        print("Predicción:\n", np.round(m.forward(X), 4))

    print("\n=== 3. ACTIVIDAD 2: FUNCIÓN SIGMOIDE ===")
    np.random.seed(42)
    mlp_sig = SimpleMLP(2, 4, 1, activation='sigmoid')
    mlp_sig.train(X, y, learning_rate=0.1, epochs=2000, print_interval=400)
    print("Predicción Sigmoide:\n", np.round(mlp_sig.forward(X), 4))

    print("\n=== 4. ACTIVIDAD 3: MLP 4 CAPAS ===")
    deep_mlp = DeepMLP4Layers(2, 4, 4, 1)
    deep_mlp.train(X, y, learning_rate=0.08, epochs=2000, print_interval=400)
    print("Predicción final 4 capas:\n", np.round(deep_mlp.forward(X), 4))