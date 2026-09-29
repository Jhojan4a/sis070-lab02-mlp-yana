# Laboratorio 04: El Perceptrón Multicapa (MLP) y Backpropagation desde Cero

* **Curso:** SIS070 - Inteligencia Artificial
* **Unidad:** Unidad II - Aprendizaje Supervisado y Redes Neuronales
* **Estudiante:** Jhojan Abel Yana Ramos
* **Repositorio:** https://github.com/Jhojan4a/sis070-lab02-mlp-yana

---

## 1. Descripción y Arquitectura del Modelo

En esta práctica se implementó una red neuronal artificial **Perceptrón Multicapa (MLP)** utilizando Python y la biblioteca **NumPy**, sin utilizar frameworks especializados de Deep Learning como TensorFlow o PyTorch.

El objetivo principal es comprender el funcionamiento interno de una red neuronal mediante la implementación desde cero de los procesos de propagación hacia adelante, cálculo de pérdida, retropropagación del error y actualización de pesos mediante descenso de gradiente.

El problema utilizado para evaluar la red neuronal es la compuerta lógica **XOR**, que representa un problema no linealmente separable.

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

La función de activación permite introducir no linealidad en la red y posibilita el aprendizaje de relaciones más complejas entre las entradas y las salidas.

#### 2. Función de Pérdida (Loss Function)

Se emplea el **Error Cuadrático Medio (MSE)**:

$$
Loss = \frac{1}{m}\sum_{i=1}^{m}(A_i^{[L]}-y_i)^2
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

* \(W\) representa los pesos.
* \(b\) representa los sesgos.
* \(\alpha\) representa la tasa de aprendizaje.
* \(dW\) y \(db\) representan los gradientes calculados mediante backpropagation.

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

Se evaluó el comportamiento del entrenamiento variando el hiperparámetro **learning_rate** utilizando valores extremos:

* `0.9`
* `0.0001`

El objetivo fue observar cómo la tasa de aprendizaje afecta la velocidad de reducción de la función de pérdida y el comportamiento final del modelo.

#### Tasa Alta: 0.9

Con `learning_rate = 0.9`, se obtuvieron los siguientes resultados:

```text
Época    0 | Pérdida (Loss): 0.511851
Época  400 | Pérdida (Loss): 0.166667
Época  800 | Pérdida (Loss): 0.166667

Loss final: 0.166667
```

Predicción:

```text
[[0.6667]
 [0.6667]
 [0.6667]
 [0.    ]]
```

La pérdida disminuyó rápidamente durante las primeras épocas y posteriormente permaneció prácticamente constante alrededor de `0.166667`.

En esta ejecución no se observó una divergencia numérica de la pérdida. Sin embargo, el modelo quedó estancado en una solución que no permitió reproducir correctamente las cuatro salidas esperadas del problema XOR.

#### Tasa Baja: 0.0001

Con `learning_rate = 0.0001`, se obtuvieron los siguientes resultados:

```text
Época    0 | Pérdida (Loss): 0.511851
Época  400 | Pérdida (Loss): 0.491092
Época  800 | Pérdida (Loss): 0.472000

Loss final: 0.463035
```

Predicciones finales:

```text
[[0.0486]
 [0.0392]
 [0.0378]
 [0.0293]]
```

En este caso, la disminución de la pérdida fue considerablemente más lenta.

Después de las épocas utilizadas en el experimento, la pérdida final fue `0.463035`, por lo que el modelo no tuvo suficiente velocidad de aprendizaje para alcanzar una solución adecuada para XOR dentro del número de épocas utilizado.

#### Conclusión de la Actividad 1

Los resultados muestran que la tasa de aprendizaje tiene una influencia directa sobre la velocidad de actualización de los parámetros.

Con `learning_rate = 0.9`, la pérdida disminuyó rápidamente, pero posteriormente se produjo un estancamiento en una solución que no resolvió completamente XOR.

Con `learning_rate = 0.0001`, las actualizaciones fueron mucho más pequeñas y la reducción de la pérdida fue lenta.

Por lo tanto, la tasa de aprendizaje debe seleccionarse considerando tanto la velocidad de aprendizaje como la capacidad del modelo para alcanzar una solución adecuada.

---

### Actividad 2: Cambio de Función de Activación (ReLU vs. Sigmoide)

Se sustituyó la función de activación **ReLU** por la función **Sigmoide**, implementando también su derivada analítica.

#### ReLU

La función ReLU se define como:

$$
ReLU(z) = \max(0,z)
$$

Su derivada es:

$$
ReLU'(z)=
\begin{cases}
1, & z > 0 \\
0, & z \leq 0
\end{cases}
$$

En el experimento base se utilizó:

```text
learning_rate = 0.1
```

Los resultados fueron:

```text
Época    0 | Pérdida (Loss): 0.511851
Época  200 | Pérdida (Loss): 0.248587
Época  400 | Pérdida (Loss): 0.220127
Época  600 | Pérdida (Loss): 0.169911
Época  800 | Pérdida (Loss): 0.166680
```

Predicciones finales:

```text
[[0.6667]
 [0.6667]
 [0.6667]
 [0.    ]]
```

Por lo tanto, bajo esta configuración, la red logró reducir progresivamente la pérdida, pero no consiguió reproducir exactamente el patrón XOR.

#### Sigmoide

La función Sigmoide se define como:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Su derivada es:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

Su valor máximo de derivada es `0.25` y disminuye cuando la activación se aproxima a 0 o 1.

En el experimento realizado se utilizó la función Sigmoide durante 2000 épocas.

Los resultados fueron:

```text
Época    0 | Pérdida (Loss): 0.871317
Época  400 | Pérdida (Loss): 0.249647
Época  800 | Pérdida (Loss): 0.249314
Época 1200 | Pérdida (Loss): 0.248983
Época 1600 | Pérdida (Loss): 0.248528
```

Predicciones finales:

```text
[[0.4648]
 [0.5054]
 [0.5017]
 [0.5312]]
```

La pérdida disminuyó desde `0.871317` hasta aproximadamente `0.248528`. Sin embargo, las predicciones permanecieron cercanas a `0.5`, por lo que la red no logró aprender correctamente el patrón XOR bajo esta configuración.

#### Comparación ReLU vs. Sigmoide

Los experimentos permiten observar diferencias en el proceso de entrenamiento.

Con ReLU, la pérdida disminuyó desde `0.511851` hasta aproximadamente `0.166680`, aunque el modelo no consiguió resolver completamente XOR.

Con Sigmoide, la pérdida disminuyó desde `0.871317` hasta aproximadamente `0.248528`, pero las predicciones finales permanecieron cercanas a `0.5`.

Por lo tanto, en las configuraciones utilizadas en este laboratorio, ninguna de las dos configuraciones de la red simple consiguió resolver completamente XOR.

Esto demuestra que el comportamiento de la red depende no solamente de la función de activación, sino también de la arquitectura, los hiperparámetros y la configuración utilizada durante el entrenamiento.

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

#### Resultados del entrenamiento

La red de cuatro capas obtuvo los siguientes resultados:

```text
Época    0 | Pérdida (Loss): 0.545536
Época  400 | Pérdida (Loss): 0.080454
Época  800 | Pérdida (Loss): 0.000047
Época 1200 | Pérdida (Loss): 0.000000
Época 1600 | Pérdida (Loss): 0.000000
```

Predicción final:

```text
[[0.]
 [1.]
 [1.]
 [0.]]
```

Estas predicciones coinciden exactamente con las salidas esperadas del problema XOR.

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

La incorporación de una segunda capa oculta permitió realizar transformaciones adicionales sobre las representaciones internas de los datos.

En este experimento, la arquitectura ampliada consiguió reducir la pérdida prácticamente hasta cero y reproducir correctamente las cuatro salidas de XOR.

---

## 4. Implementación

La implementación fue realizada desde cero utilizando Python y NumPy, sin utilizar frameworks especializados de aprendizaje profundo como TensorFlow o PyTorch.

Las principales clases desarrolladas son:

### SimpleMLP

Implementa una red neuronal multicapa básica para resolver el problema XOR.

Sus principales procesos son:

* Inicialización de pesos.
* Propagación hacia adelante.
* Función de activación.
* Cálculo de pérdida.
* Retropropagación.
* Cálculo de gradientes.
* Actualización de pesos mediante descenso de gradiente.
* Predicción de resultados.

### DeepMLP4Layers

Implementa una arquitectura más profunda:

```text
2 → 4 → 4 → 1
```

Esta versión permite analizar el comportamiento del algoritmo de backpropagation cuando existen múltiples capas ocultas.

---

## 5. Problema XOR

El conjunto de datos utilizado corresponde a la compuerta lógica XOR:

| **Entrada X1** | **Entrada X2** | **Salida esperada** |
| -------------- | -------------- | ------------------- |
| 0              | 0              | 0                   |
| 0              | 1              | 1                   |
| 1              | 0              | 1                   |
| 1              | 1              | 0                   |

El problema XOR no puede ser separado correctamente mediante una única frontera lineal.

Por esta razón, se requiere una arquitectura con capas ocultas y funciones de activación no lineales para representar correctamente la relación entre las entradas y la salida.

---

## 6. Tecnologías Utilizadas

* **Python 3**
* **NumPy**
* **Git**
* **GitHub**
* **Visual Studio Code**

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

* El efecto de diferentes tasas de aprendizaje.
* La diferencia entre las funciones de activación ReLU y Sigmoide.
* El comportamiento de una arquitectura neuronal más profunda.
* El proceso de propagación hacia adelante.
* El proceso de retropropagación.
* El cálculo y propagación de gradientes.
* La actualización de pesos mediante descenso de gradiente.

### Resumen de resultados

| **Experimento**   | **Loss final aproximado** | **Resultado**                 |
| ----------------- | ------------------------: | ----------------------------- |
| ReLU, `lr=0.1`    |                `0.166680` | No resuelve completamente XOR |
| ReLU, `lr=0.9`    |                `0.166667` | No resuelve completamente XOR |
| ReLU, `lr=0.0001` |                `0.463035` | Aprendizaje muy lento         |
| Sigmoide          |                `0.248528` | No resuelve XOR               |
| MLP 4 capas       |                `0.000000` | Resuelve XOR correctamente    |

El resultado obtenido con la arquitectura de cuatro capas fue:

```text
2 → 4 → 4 → 1
```

que produjo las predicciones:

```text
[0, 1, 1, 0]
```

Estas predicciones coinciden con las salidas esperadas del problema XOR.

---

## 9. Conclusiones

1. El Perceptrón Multicapa permite resolver problemas que presentan relaciones no lineales como el problema XOR.
2. La tasa de aprendizaje tiene una influencia directa sobre la velocidad y el comportamiento del proceso de entrenamiento.
3. Una tasa de aprendizaje demasiado baja puede hacer que el aprendizaje sea excesivamente lento.
4. Una tasa de aprendizaje elevada puede producir una reducción rápida de la pérdida, pero no garantiza que el modelo encuentre una solución adecuada.
5. La función de activación utilizada influye directamente en la propagación del gradiente y en el comportamiento de la convergencia.
6. En el experimento realizado, la red simple con ReLU no consiguió resolver completamente XOR, aunque logró reducir progresivamente la función de pérdida.
7. En el experimento con Sigmoide, la pérdida disminuyó, pero las predicciones permanecieron cercanas a `0.5`, por lo que tampoco se resolvió correctamente XOR bajo esta configuración.
8. La incorporación de una segunda capa oculta mediante la arquitectura `2 → 4 → 4 → 1` permitió obtener una pérdida prácticamente igual a cero y las predicciones correctas `[0, 1, 1, 0]`.
9. La implementación de backpropagation desde cero permite comprender cómo se calculan y propagan los gradientes a través de las diferentes capas de una red neuronal.
10. Los experimentos permitieron comprobar que la arquitectura, la función de activación, la tasa de aprendizaje y el número de épocas influyen conjuntamente en el proceso de aprendizaje de una red neuronal.

---

## 10. Autor

**Jhojan Abel Yana Ramos**

Curso: **SIS070 - Inteligencia Artificial**

Unidad II - Aprendizaje Supervisado y Redes Neuronales

Repositorio:

https://github.com/Jhojan4a/sis070-lab02-mlp-yana
