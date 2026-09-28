# Laboratorio 04: Perceptrón Multicapa (MLP) y Backpropagation

## 1. Estructura
- `src/mlp_implementation.py`: Implementación de red neuronal en NumPy.
- `requirements.txt`: Dependencias requeridas.

## 2. Análisis de Actividades
- **Actividad 1 (Tasa de aprendizaje):** Con lr=0.9 el paso es muy brusco provocando oscilaciones, mientras que con lr=0.0001 la convergencia es prácticamente nula en 1000 épocas.
- **Actividad 2 (ReLU vs Sigmoide):** ReLU converge más rápido gracias a su gradiente constante (1 para z>0). Sigmoide requiere más épocas debido a la atenuación de gradientes en sus extremos.
- **Actividad 3 (4 Capas):** Se implementó una segunda capa oculta encadenando derivadas parciales con éxito.