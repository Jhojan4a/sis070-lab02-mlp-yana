# Laboratorio 04: El Perceptrón Multicapa (MLP) y Backpropagation desde Cero

- **Curso:** SIS070 - Inteligencia Artificial
- **Unidad:** Unidad II - Aprendizaje Supervisado y Redes Neuronales
- **Estudiante:** Jhojan Abel Yana Ramos
- **Repositorio:** https://github.com/Jhojan4a/sis070-lab02-mlp-yana

---

## 1. Descripción y Arquitectura del Modelo

En esta práctica se implementó una red neuronal artificial Perceptrón Multicapa (MLP) en Python puro utilizando únicamente la biblioteca NumPy, resolviendo el clásico problema no linealmente separable de la compuerta lógica XOR.

### Flujo Matemático y Algorítmico
1. **Propagación hacia adelante (Forward Propagation):**
   - Para cada capa $l$, la combinación lineal se calcula como:  
     $$Z^{[l]} = A^{[l-1]} \cdot W^{[l]} + b^{[l]}$$
   - Se aplica una función de activación no lineal $A^{[l]} = g(Z^{[l]})$ que deforma el espacio para clasificar datos no separables linealmente.
2. **Función de Pérdida (Loss Function):**
   - Se emplea el Error Cuadrático Medio (MSE):  
     $$Loss = \frac{1}{m} \sum (A^{[L]} - y)^2$$
3. **Retropropagación (Backpropagation):**
   - Aplicando la regla de la cadena, el error cometido en la salida se propaga hacia atrás, calculando las derivadas parciales respecto a los pesos ($\frac{\partial Loss}{\partial W}$) y los sesgos ($\frac{\partial Loss}{\partial b}$).
4. **Descenso de Gradiente:**
   - Parámetros actualizados de forma iterativa:  
     $$W = W - \alpha \cdot dW \quad \text{y} \quad b = b - \alpha \cdot db$$

---

## 2. Estructura del Repositorio

```text
sis070-lab02-mlp-yana/
├── src/
│   └── mlp_implementation.py   # Script con clases SimpleMLP, DeepMLP4Layers y experimentos
├── README.md                   # Documentación, fundamentación teórica y reporte de resultados
└── requirements.txt            # Dependencias del entorno (numpy)