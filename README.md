# CRISP-ML Image Classifier: Feibert vs Tomas

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0%2B-FF4B4B?logo=streamlit&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0%2B-FF6F00?logo=tensorflow&logoColor=white)](#)
[![Keras](https://img.shields.io/badge/Keras-2.15.0%2B-D00000?logo=keras&logoColor=white)](#)
[![CRISP-ML](https://img.shields.io/badge/Methodology-CRISP--ML-8A2BE2)](#)

Este proyecto convierte un modelo de clasificación de imágenes Keras (entrenado para identificar y clasificar los rostros de **Feibert/David** y **Tomas**) en un sistema de producción robusto estructurado bajo la metodología industrial **CRISP-ML(Q)**.

Consiste en dos componentes principales:
1. **Landing Page (`index.html` & `style.css`)**: Un sitio web interactivo moderno y responsivo con estética dark mode y glassmorphic que describe detalladamente las fases de CRISP-ML en el proyecto.
2. **Dashboard Interactivo de Inferencia (`app.py`)**: Una interfaz en Streamlit que carga el modelo en tiempo real, procesa fotos cargadas o capturas de webcam, permite ajustar el umbral de confianza y simula la recopilación de muestras para el ciclo de mantenimiento.

---

## Estructura del Proyecto

```text
├── Proyectokeras/
│   └── Keras_Tomas_David/
│       ├── keras_model.h5     # Modelo convolucional exportado de Keras
│       └── labels.txt         # Clases del modelo (0 Tomas, 1 David)
├── .agents/
│   └── skills/
│       └── crisp-ml-classifier/
│           └── SKILL.md       # Guía para el agente de desarrollo y retraining
├── index.html                 # Landing page principal
├── style.css                  # Estilos glassmorphic dark-theme
├── app.py                     # Dashboard de Streamlit
├── requirements.txt           # Dependencias de Python
└── README.md                  # Documentación del proyecto (este archivo)
```

---

## Requisitos Previos

- Python 3.8, 3.9, 3.10 o 3.11.
- Conectividad a internet para dependencias y carga de fuentes.
- Una cámara web (opcional, para usar la entrada de video en vivo).

---

## Instalación y Ejecución

### 1. Clonar o descargar el espacio de trabajo
Asegúrese de estar ubicado en la raíz del proyecto:
```bash
cd c:\Users\ASUS\Downloads\Proyectokeras-20260821T123847Z-1-001
```

### 2. Instalar dependencias
Se recomienda utilizar un entorno virtual (venv):
```bash
python -m venv venv
venv\Scripts\activate      # En Windows (CMD)
# o
.\venv\Scripts\Activate.ps1 # En Windows (PowerShell)

pip install -r requirements.txt
```

### 3. Iniciar el Dashboard de Inferencia (Streamlit)
Ejecute el siguiente comando para levantar el servidor local de Streamlit:
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en su navegador en `http://localhost:8501`.

### 4. Abrir la Landing Page Metodológica
Simplemente haga doble clic en [index.html](file:///c:/Users/ASUS/Downloads/Proyectokeras-20260821T123847Z-1-001/index.html) o ábralo en su navegador web favorito. Esta página le servirá como explicación teórica interactiva y posee accesos directos hacia el dashboard.

---

## Mapeo Metodológico CRISP-ML(Q)

El desarrollo del proyecto está enmarcado en el estándar de calidad CRISP-ML:
1. **Comprensión del Negocio**: Clasificar con precisión superior al 90% con tiempos mínimos de latencia para evitar retrasos de inferencia.
2. **Preparación de Datos**: Redimensión de imágenes a $224 \times 224$ píxeles y escalado del rango numérico a $[-1.0, 1.0]$.
3. **Modelado**: Uso de redes neuronales convolucionales profundas entrenadas vía Keras.
4. **Evaluación**: Ajuste de un umbral de confianza personalizable para evitar predicciones dudosas.
5. **Despliegue**: Inferencia en vivo desde cámara web o subida local con Streamlit.
6. **Mantenimiento**: Captura de logs de acierto/error que alimentan el pool de retraining para mitigar la deriva de datos (data drift).
