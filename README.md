# Introducción a los Algoritmos de Aprendizaje (IAA)

> **Autores:** Alicia Muriel Fernández y Lucía Cañero Moslero — *Estudiante de Ingeniería Informática, Universidad de Córdoba*

---

## Descripción del Proyecto

Repositorio que contiene las prácticas, implementación de modelos y análisis experimental correspondientes a la asignatura de **Introducción a los Algoritmos de Aprendizaje (IAA)** del Grado en Ingeniería Informática de la Universidad de Córdoba (UCO).

El objetivo principal es el estudio, implementación y evaluación de algoritmos de **Aprendizaje Automático (Machine Learning)** tanto supervisado como no supervisado, abarcando desde la preparación y preprocesamiento de datos hasta el ajuste de hiperparámetros y evaluación de modelos.

---

## Contenidos y Algoritmos Implementados

* **Preprocesamiento de Datos:**
  * Limpieza de datos, tratamiento de valores nulos y escalado/normalización de características (*StandardScaler*, *MinMaxScaler*).
  * Selección y reducción de dimensionalidad de características.

* **Aprendizaje Supervisado:**
  * **Clasificación:** Regresión Logística, Árboles de Decisión, $k$-Nearest Neighbors ($k$-NN), Support Vector Machines (SVM) y Métodos Ensamble (Random Forest, Boosting).
  * **Regresión:** Regresión Lineal y Polinómica.

* **Aprendizaje No Supervisado:**
  * Algoritmos de Agrupamiento (*Clustering*) como $k$-Means o Agrupamiento Jerárquico.

* **Evaluación y Validación de Modelos:**
  * Validación Cruzada (*Cross-Validation*).
  * Métricas de rendimiento: Matriz de confusión, Precisión (*Accuracy*), Sensibilidad (*Recall*), *F1-Score*, Curvas ROC-AUC.
  * Optimización de hiperparámetros mediante *Grid Search* y *Random Search*.

---

## Tecnologías y Librerías Utilizadas

* **Lenguaje:** Python 3.x
* **Análisis y Manipulación de Datos:** NumPy, Pandas
* **Machine Learning:** Scikit-Learn
* **Visualización:** Matplotlib, Seaborn
* **Entorno de Desarrollo:** Jupyter Notebooks / VS Code

---

## 📁 Estructura del Repositorio

```text
├── practicas/        # Notebooks (.ipynb) o scripts (.py) con las resoluciones
├── datasets/         # Conjuntos de datos utilizados para entrenamiento y pruebas
├── doc/              # Informes técnicos y documentación explicativa
└── README.md
```

---

## Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/aliciamurielf/IAA.git
   cd IAA
   ```

2. **Crear y activar un entorno virtual (opcional pero recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/macOS
   # venv\Scripts\activate   # En Windows
   ```

3. **Instalar dependencias necesarias:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   ```

4. **Ejecutar los cuadernos Jupyter:**
   ```bash
   jupyter notebook
   ```
