#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
------
Aplicación Streamlit para análisis fractal:
- Tab 1: Análisis de dimensionalidad fractal desde imagen
- Tab 2: Interpolación fractal de series temporales con huecos
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import io

# Importar funciones de los módulos locales
from dimensionalidad_utils import (
    load_points_from_image,
    box_counting_dimension,
)
from fractal_interpolation_utils import (
    fractal_interpolate_series,
)

st.set_page_config(page_title="Análisis Fractal", layout="wide")

st.title("🔬 Análisis Fractal")

tab1, tab2 = st.tabs(["📐 Dimensionalidad Fractal", "📈 Interpolación Fractal"])

# ========== TAB 1: Dimensionalidad Fractal ==========
with tab1:
    st.header("Análisis de Dimensionalidad Fractal")
    st.markdown("Sube una imagen de un fractal para estimar su dimensión usando box-counting (Minkowski-Bouligand)")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_image = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg", "tiff", "bmp"], key="img")

        if uploaded_image is not None:
            # Mostrar imagen original
            img = Image.open(uploaded_image)
            st.image(img, caption="Imagen cargada", use_container_width=True)

            # Parámetros
            st.subheader("Parámetros")
            threshold = st.slider("Umbral de binarización", 0.0, 1.0, 0.5, 0.05)
            invert = st.checkbox("Invertir máscara (fondo claro)", value=False)

            col_a, col_b = st.columns(2)
            with col_a:
                eps_base = st.number_input("Base para ε (escalas)", min_value=2.0, max_value=10.0, value=2.0, step=0.5)
            with col_b:
                eps_range = st.text_input("Rango de potencias k (ej: 2:9)", value="2:9")

            min_points_fit = st.slider("Mínimo de escalas para ajuste", 3, 10, 5)

            if st.button("Calcular Dimensión", type="primary"):
                with st.spinner("Procesando..."):
                    try:
                        # Guardar temporalmente la imagen
                        temp_path = "/tmp/temp_fractal.png"
                        img.save(temp_path)

                        # Cargar puntos
                        pts = load_points_from_image(temp_path, threshold=threshold, invert=invert)

                        if pts.shape[0] < 2:
                            st.error("Muy pocos píxeles activos tras el umbral. Ajusta los parámetros.")
                        else:
                            # Construir lista de escalas
                            k_min, k_max = map(int, eps_range.split(":"))
                            eps_list = [eps_base ** (-k) for k in range(k_min, k_max + 1)]

                            # Calcular dimensión
                            D, fit, series = box_counting_dimension(
                                pts,
                                eps_list=eps_list,
                                auto_window=True,
                                min_points_fit=min_points_fit,
                                return_series=True
                            )

                            with col2:
                                st.subheader("Resultados")
                                st.metric("Dimensión Fractal (D)", f"{D:.4f}")
                                st.metric("Coeficiente de correlación (R)", f"{fit.rvalue:.3f}")
                                st.metric("Escalas usadas", len(fit.idx_used))

                                # Gráfico log-log
                                if series is not None:
                                    xs, ys = series
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    ax.scatter(xs, ys, s=50, alpha=0.6, label="Datos (todas escalas)")
                                    xw = xs[fit.idx_used]
                                    ax.plot(xw, fit.slope * xw + fit.intercept,
                                           linewidth=2.5, color='red',
                                           label=f"Ajuste (R={fit.rvalue:.3f})")
                                    ax.set_xlabel("log(1/ε)", fontsize=12)
                                    ax.set_ylabel("log N(ε)", fontsize=12)
                                    ax.set_title("Box-counting", fontsize=14)
                                    ax.grid(True, which="both", ls="--", alpha=0.3)
                                    ax.legend()
                                    st.pyplot(fig)
                                    plt.close()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# ========== TAB 2: Interpolación Fractal ==========
with tab2:
    st.header("Interpolación Fractal de Series Temporales")
    st.markdown("Sube un CSV con columnas `x` e `y` (puede contener NaN en `y`) para reconstruir usando FIF")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_csv = st.file_uploader("Selecciona un archivo CSV", type=["csv"], key="csv")

        if uploaded_csv is not None:
            # Leer CSV
            df = pd.read_csv(uploaded_csv)

            st.subheader("Datos cargados")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Total de filas: {len(df)}")

            # Validar columnas
            if df.shape[1] < 2:
                st.error("El CSV debe tener al menos 2 columnas (x, y)")
            else:
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float)

                known_mask = np.isfinite(x) & np.isfinite(y)
                n_known = np.sum(known_mask)
                n_missing = len(y) - n_known

                st.info(f"Puntos conocidos: {n_known} | Puntos faltantes: {n_missing}")

                # Parámetros
                st.subheader("Parámetros de interpolación")
                alpha = st.slider("Factor de rugosidad α (|α| < 1)", 0.0, 0.99, 0.3, 0.05,
                                 help="Controla la rugosidad de la curva fractal. Valores menores = más suave")
                n_refinements = st.slider("Iteraciones de refinamiento", 5, 20, 12)

                if st.button("Interpolar", type="primary"):
                    with st.spinner("Interpolando..."):
                        try:
                            # Reconstruir serie
                            x_hat, y_hat = fractal_interpolate_series(
                                x, y,
                                alpha=alpha,
                                x_out=x,
                                n_refinements=n_refinements
                            )

                            with col2:
                                st.subheader("Resultados")

                                # Gráfico
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(x, y, s=30, alpha=0.5, label="Observado (con NaN)", color='blue')
                                ax.plot(x_hat, y_hat, lw=2, label=f"FIF (α={alpha})", color='red')
                                ax.set_xlabel("x", fontsize=12)
                                ax.set_ylabel("y", fontsize=12)
                                ax.set_title("Interpolación Fractal", fontsize=14)
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close()

                                # Descargar resultados
                                result_df = pd.DataFrame({"x": x_hat, "y_reconstructed": y_hat})
                                csv_buffer = io.StringIO()
                                result_df.to_csv(csv_buffer, index=False)

                                st.download_button(
                                    label="📥 Descargar CSV reconstruido",
                                    data=csv_buffer.getvalue(),
                                    file_name="interpolacion_fractal.csv",
                                    mime="text/csv"
                                )

                        except Exception as e:
                            st.error(f"Error durante la interpolación: {str(e)}")

st.markdown("---")
st.caption("Aplicación de análisis fractal | Usa los métodos de dimensionalidad.py y fractal_interpolation.py")
