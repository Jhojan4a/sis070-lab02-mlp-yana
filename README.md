# Laboratorio 04: El Perceptrón Multicapa (MLP) y Backpropagation desde Cero

- **Curso:** SIS070 - Inteligencia Artificial
- **Unidad:** Unidad II - Aprendizaje Supervisado y Redes Neuronales
- **Estudiante:** Jhojan Abel Yana Ramos
- **Repositorio:** https://github.com/Jhojan4a/sis070-lab02-mlp-yana

---

## 1. Descripción y Arquitectura del Modelo

En esta práctica se implementó una red neuronal artificial **Perceptrón Multicapa (MLP)** en Python puro utilizando únicamente la biblioteca **NumPy**, resolviendo el clásico problema no linealmente separable de la compuerta lógica **XOR**.

### Flujo Matemático y Algorítmico

#### 1. Propagación hacia adelante (Forward Propagation)

Para cada capa \(l\), la combinación lineal se calcula como:

$$
Z^{[l]} = A^{[l-1]} \cdot W^{[l]} + b^{[l]}
$$

Posteriormente, se aplica una función de activación no lineal:

$$
A^{[l]} = g(Z^{[l]})
$$

La función de activación permite que la red pueda aprender relaciones no lineales y resolver problemas como XOR.

#### 2. Función de Pérdida (Loss Function)

Se emplea el **Error Cuadrático Medio (MSE)**:

$$
Loss = \frac{1}{m}\sum_{i=1}^{m}(A^{[L]}_i-y_i)^2
$$

Esta función permite medir la diferencia entre las predicciones realizadas por la red y los valores reales esperados.

#### 3. Retropropagación (Backpropagation)

Mediante la regla de la cadena, el error cometido en la salida se propaga hacia las capas anteriores.

Se calculan las derivadas parciales respecto a los pesos y sesgos:

$$
\frac{\partial Loss}{\partial W}
$$

$$
\frac{\partial Loss}{\partial b}
$$

Estos gradientes permiten determinar cómo deben modificarse los parámetros de la red durante el entrenamiento.

#### 4. Descenso de Gradiente

Los parámetros de la red se actualizan iterativamente mediante:

$$
W = W - \alpha \cdot dW
$$

$$
b = b - \alpha \cdot db
$$

Donde:

- \(W\) representa los pesos.
- \(b\) representa los sesgos.
- \(\alpha\) representa la tasa de aprendizaje.
- \(dW\) y \(db\) representan los gradientes calculados mediante backpropagation.

---

## 2. Estructura del Repositorio

```text
sis070-lab02-mlp-yana/
│
├── src/
│   └── mlp_implementation.py
│       # Implementación de las clases SimpleMLP y DeepMLP4Layers
│       # y ejecución de los diferentes experimentos.
│
├── README.md
│   # Documentación, fundamentación teórica y análisis de resultados.
│
└── requirements.txt
    # Dependencias necesarias para ejecutar el proyecto.
```

---

## 3. Reporte de Actividades Prácticas y Análisis de Resultados

### Actividad 1: Modificación de Hiperparámetros (Learning Rate)

Se evaluó el comportamiento del entrenamiento variando el hiperparámetro **learning_rate** utilizando valores extremos.

#### Tasa Alta: 0.9

- **Comportamiento:** Las actualizaciones de los pesos realizan saltos excesivamente grandes sobre la superficie de error.
- **Impacto en Loss:** Puede provocar oscilaciones importantes en el valor de pérdida. En modelos más complejos puede producir inestabilidad o dificultar la convergencia.

#### Tasa Baja: 0.0001

- **Comportamiento:** La magnitud de las actualizaciones realizadas durante cada época es muy pequeña.
- **Impacto en Loss:** En 1000 épocas, la función de costo presenta una reducción muy lenta, por lo que el modelo puede no alcanzar una solución adecuada para el problema XOR dentro del número de épocas establecido.

#### Conclusión

Para este experimento, valores intermedios entre **0.05 y 0.1** proporcionan un equilibrio adecuado entre velocidad de aprendizaje y estabilidad durante el entrenamiento.

---

### Actividad 2: Cambio de Función de Activación (ReLU vs. Sigmoide)

Se sustituyó la función de activación **ReLU** por la función **Sigmoide**, implementando también su derivada analítica.

#### Comparativa de Convergencia

**ReLU:**

La función ReLU se define como:

$$
ReLU(z) = \max(0,z)
$$

Su derivada es:

$$
ReLU'(z)=
\begin{cases}
1 & \text{si } z>0\\
0 & \text{si } z\leq0
\end{cases}
$$

En las zonas donde \(z>0\), el gradiente puede propagarse directamente, permitiendo realizar actualizaciones eficientes durante el entrenamiento.

En el experimento realizado, la red logró resolver el problema XOR en menos de **800 épocas**.

**Sigmoide:**

La función sigmoide se define como:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Su derivada es:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

Su valor máximo de derivada es 0.25 y disminuye cuando la activación se aproxima a 0 o 1.

En el experimento realizado, la utilización de Sigmoide requirió aproximadamente entre **1500 y 2000 épocas** para alcanzar una estabilización comparable.

---

### Actividad 3: Ampliación Arquitectural - Red Neuronal de 4 Capas

Se implementó la clase **DeepMLP4Layers** utilizando la siguiente arquitectura:

```text
Entrada
  2 neuronas
      │
      ▼
Capa Oculta 1
  4 neuronas
      │
      ▼
Capa Oculta 2
  4 neuronas
      │
      ▼
Capa de Salida
  1 neurona
```

La arquitectura puede representarse de forma resumida como:

```text
Entrada (2) → Oculta 1 (4) → Oculta 2 (4) → Salida (1)
```

#### Cálculo de Gradientes en Cascada

La retropropagación se realizó en diferentes etapas aplicando la regla de la cadena.

**1. Error en la capa de salida:**

```python
error3 = (output - y) / m
```

**2. Gradiente propagado hacia la segunda capa oculta:**

```python
error2 = np.dot(error3, W3.T) * relu_derivative(Z2)
```

**3. Gradiente propagado hacia la primera capa oculta:**

```python
error1 = np.dot(error2, W2.T) * relu_derivative(Z1)
```

Este procedimiento permite que el error calculado en la salida sea propagado progresivamente hacia las capas anteriores, permitiendo actualizar todos los pesos y sesgos de la red.

La red de mayor profundidad permite realizar una transformación progresiva de los datos mediante diferentes representaciones internas.

---

## 4. Implementación

La implementación fue realizada desde cero utilizando Python y NumPy, sin utilizar frameworks especializados de aprendizaje profundo como TensorFlow o PyTorch.

Las principales clases desarrolladas son:

### SimpleMLP

Implementa una red neuronal multicapa básica para resolver el problema XOR.

Sus principales procesos son:

- Inicialización de pesos.
- Propagación hacia adelante.
- Función de activación.
- Cálculo de pérdida.
- Retropropagación.
- Actualización de pesos mediante descenso de gradiente.
- Predicción de resultados.

### DeepMLP4Layers

Implementa una arquitectura más profunda:

```text
2 → 4 → 4 → 1
```

Esta versión permite analizar el comportamiento del algoritmo de backpropagation cuando existen múltiples capas ocultas.

---

## 5. Problema XOR

El conjunto de datos utilizado corresponde a la compuerta lógica XOR:

| Entrada X1 | Entrada X2 | Salida esperada |
|------------|------------|-----------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

El problema XOR no puede ser separado correctamente mediante una única frontera lineal.

Por esta razón, se requiere una arquitectura con capas ocultas y funciones de activación no lineales.

---

## 6. Tecnologías Utilizadas

- **Python 3**
- **NumPy**
- **Git**
- **GitHub**
- **Visual Studio Code**

No se utilizaron frameworks de Deep Learning para la implementación principal.

---

## 7. Instalación y Ejecución

### Clonar el repositorio

```bash
git clone https://github.com/Jhojan4a/sis070-lab02-mlp-yana.git
```

Ingresar al directorio:

```bash
cd sis070-lab02-mlp-yana
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Ejecutar el programa

```bash
python src/mlp_implementation.py
```

---

## 8. Resultados

La implementación permite observar el proceso de aprendizaje de la red neuronal mediante la reducción progresiva de la función de pérdida.

Los experimentos realizados permitieron analizar:

- El efecto de diferentes tasas de aprendizaje.
- La diferencia entre las funciones de activación ReLU y Sigmoide.
- El comportamiento de una arquitectura neuronal más profunda.
- El proceso de propagación hacia adelante.
- El proceso de retropropagación.
- La actualización de pesos mediante descenso de gradiente.

La red consigue aprender el comportamiento de la compuerta XOR mediante la combinación de capas ocultas, funciones de activación no lineales y el algoritmo de backpropagation.

---

## 9. Conclusiones

1. El Perceptrón Multicapa permite resolver problemas que no son linealmente separables, como XOR.

2. La tasa de aprendizaje tiene una influencia directa sobre la velocidad y estabilidad del proceso de entrenamiento.

3. Una tasa de aprendizaje demasiado elevada puede producir oscilaciones, mientras que una tasa demasiado baja puede hacer que el aprendizaje sea excesivamente lento.

4. La función de activación utilizada influye directamente en la propagación del gradiente y en la velocidad de convergencia.

5. La implementación de backpropagation desde cero permite comprender el funcionamiento interno del entrenamiento de las redes neuronales.

6. La incorporación de capas ocultas adicionales permite realizar transformaciones progresivas de los datos y estudiar el comportamiento de redes con mayor profundidad.

---

## 10. Autor

**Jhojan Abel Yana Ramos**

Curso: **SIS070 - Inteligencia Artificial**

Unidad II - Aprendizaje Supervisado y Redes Neuronales

Repositorio:

https://github.com/Jhojan4a/sis070-lab02-mlp-yana