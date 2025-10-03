# 🔬 Aplicación de Análisis Fractal

Aplicación web interactiva con Streamlit para análisis fractal, que incluye:
- **Estimación de dimensión fractal** desde imágenes usando box-counting
- **Interpolación fractal** de series temporales con huecos (FIF)

## 📋 Requisitos

```bash
pip install streamlit numpy pandas pillow matplotlib
```

## 🚀 Uso

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

## 📐 Tab 1: Dimensionalidad Fractal

Sube una imagen de un fractal (PNG, JPG, TIFF, etc.) y obtén su dimensión fractal usando el método de box-counting (Minkowski–Bouligand).

### Parámetros ajustables:
- **Umbral de binarización** (0.0 - 1.0): Controla qué píxeles se consideran "activos"
- **Invertir máscara**: Útil cuando el fondo es claro y el fractal es oscuro
- **Base para ε**: Base para las escalas (2.0 por defecto, 3.0 para Cantor/Koch)
- **Rango de potencias**: Define las escalas como `base^(-k)` para k en el rango
- **Mínimo de escalas**: Número mínimo de escalas para el ajuste log-log

### Salida:
- **Dimensión Fractal (D)**: Valor estimado de la dimensión
- **Coeficiente R**: Calidad del ajuste lineal
- **Gráfico log-log**: Visualización del ajuste y las escalas usadas

### Ejemplos de dimensiones esperadas:
- Línea recta: D ≈ 1.0
- Curva de Koch: D ≈ 1.26
- Triángulo de Sierpinski: D ≈ 1.58
- Conjunto de Cantor: D ≈ 0.63
- Superficie: D ≈ 2.0

## 📈 Tab 2: Interpolación Fractal

Sube un archivo CSV con columnas `x` e `y` (puede contener NaN en `y`) para reconstruir los valores faltantes usando Funciones de Interpolación Fractal (FIF).

### Formato del CSV:
```csv
x,y
0.0,1.2
0.1,1.5
0.2,NaN
0.3,NaN
0.4,2.1
...
```

### Parámetros ajustables:
- **α (alpha)**: Factor de rugosidad (0.0 - 0.99)
  - 0.0 - 0.3: Interpolación suave
  - 0.3 - 0.6: Rugosidad moderada
  - 0.6 - 0.9: Altamente fractal
- **Iteraciones**: Número de refinamientos (5-20, default 12)

### Salida:
- **Gráfico**: Visualización de datos observados vs. interpolados
- **CSV reconstruido**: Descarga los datos interpolados

## 📚 Módulos

### `dimensionalidad_utils.py`
Funciones para estimación de dimensión fractal:
- `load_points_from_image()`: Carga imagen y extrae puntos activos
- `box_counting_dimension()`: Calcula dimensión fractal por box-counting
- `box_count()`: Cuenta celdas ocupadas en una grilla
- Clases auxiliares: `FitResult` para ajustes lineales

### `fractal_interpolation_utils.py`
Funciones para interpolación fractal (FIF):
- `fractal_interpolate_series()`: Función principal para interpolar series con huecos
- `build_fif_params()`: Construye parámetros del sistema de funciones afines (IFS)
- `fif_evaluate()`: Evalúa la FIF en puntos específicos
- `chaos_game_sample()`: Muestreo ilustrativo usando chaos game
- Clases auxiliares: `FIFParams` para parámetros de la FIF

## 🔬 Base Teórica

### Box-Counting (Dimensión de Minkowski–Bouligand)
El método cuenta cuántas cajas de tamaño ε se necesitan para cubrir el conjunto:

```
N(ε) ~ (1/ε)^D
```

Donde D es la dimensión fractal. Se estima mediante regresión lineal en espacio log-log:

```
log N(ε) ~ D · log(1/ε)
```

### Fractal Interpolation Functions (FIF)
Sistema de funciones afines iteradas (IFS) de Barnsley-Elton-Hardin que interpola puntos dados:

Para cada intervalo i:
```
w_i: x' = a_i·x + b_i
     y' = α_i·y + p_i·x + q_i
```

El parámetro α_i (|α| < 1) controla la contracción vertical y la rugosidad de la curva resultante.

## 📖 Referencias

- Barnsley, M. F. (1986). "Fractal functions and interpolation"
- Falconer, K. (2003). "Fractal Geometry: Mathematical Foundations and Applications"
- Mandelbrot, B. (1982). "The Fractal Geometry of Nature"

## 🛠️ Desarrollo

Este módulo es independiente y contiene todas las funcionalidades necesarias. Las funciones están completamente documentadas con docstrings tipo Google/NumPy.

### Estructura:
```
app/
├── README.md
├── app.py                          # Aplicación Streamlit
├── dimensionalidad_utils.py        # Módulo de dimensión fractal
└── fractal_interpolation_utils.py  # Módulo de interpolación fractal
```
