#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fractal_interpolation_utils.py
-------------------------------
Interpolación fractal (Fractal Interpolation Function, FIF de Barnsley-Elton-Hardin)
para reconstrucción de series temporales con huecos.

Características:
- Construye una FIF sobre puntos (x, y) conocidos (sin NaN).
- Control de rugosidad con un α global o una lista α_i por tramo (|α| < 1).
- Evaluación determinista por ecuación funcional (rápida y estable).
- Opción 'chaos game' para muestrear la curva fractal (ilustrativo).

Basado en fractal_interpolation.py original (manteniendo toda la funcionalidad).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import numpy as np


# -----------------------------
# Utilidades y validaciones
# -----------------------------

def _ensure_sorted_unique(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ordena por x y colapsa duplicados promediando y.

    Args:
        x: Coordenadas x
        y: Coordenadas y

    Returns:
        Tupla (x_sorted, y_averaged) con duplicados colapsados
    """
    idx = np.argsort(x)
    x = np.asarray(x, float)[idx]
    y = np.asarray(y, float)[idx]

    # colapsar duplicados de x (si los hay) promediando y
    unique_x, inv = np.unique(x, return_inverse=True)
    if len(unique_x) == len(x):
        return x, y

    y_acc = np.zeros_like(unique_x, dtype=float)
    counts = np.zeros_like(unique_x, dtype=int)
    for i, k in enumerate(inv):
        y_acc[k] += y[i]
        counts[k] += 1
    y_avg = y_acc / np.maximum(counts, 1)

    return unique_x, y_avg


def _clean_known_points(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtra NaN/Inf y asegura al menos 2 puntos.

    Args:
        x: Coordenadas x (pueden contener NaN/Inf)
        y: Coordenadas y (pueden contener NaN/Inf)

    Returns:
        Tupla (x_clean, y_clean) con puntos válidos y únicos

    Raises:
        ValueError: Si quedan menos de 2 puntos válidos
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xk = np.asarray(x, float)[mask]
    yk = np.asarray(y, float)[mask]

    if xk.size < 2:
        raise ValueError("Se necesitan al menos dos puntos (x,y) finitos para construir la FIF.")

    xk, yk = _ensure_sorted_unique(xk, yk)

    if xk.size < 2:
        raise ValueError("Tras depurar duplicados, quedan menos de 2 puntos.")

    return xk, yk


# ------------------------------------------
# Parámetros del sistema de funciones afines
# ------------------------------------------

@dataclass
class FIFParams:
    """
    Parámetros de la Función de Interpolación Fractal (FIF).

    Para cada intervalo i=1..N, definimos w_i:
    - x' = a_i * x + b_i
    - y' = α_i * y + p_i * x + q_i

    Attributes:
        a: Coeficientes de escala horizontal por intervalo
        b: Coeficientes de traslación horizontal por intervalo
        alpha: Coeficientes de contracción vertical por intervalo (|α| < 1)
        p: Coeficientes de pendiente para la componente y
        q: Coeficientes de desplazamiento para la componente y
        x0: Primer nodo x
        xN: Último nodo x
        y0: Primer nodo y
        yN: Último nodo y
        x_nodes: Array con todos los nodos x [x0, x1, ..., xN]
        y_nodes: Array con todos los nodos y [y0, y1, ..., yN]
    """
    a: np.ndarray
    b: np.ndarray
    alpha: np.ndarray
    p: np.ndarray
    q: np.ndarray
    x0: float
    xN: float
    y0: float
    yN: float
    x_nodes: np.ndarray
    y_nodes: np.ndarray


def build_fif_params(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float | Iterable[float] = 0.3
) -> FIFParams:
    """
    Construye los parámetros de la FIF (Barnsley–Elton–Hardin).

    Fórmulas:
      a_i = (x_i - x_{i-1}) / (x_N - x_0)
      b_i = (x_{i-1} * x_N - x_i * x_0) / (x_N - x_0)
      p_i = (y_i - y_{i-1} - α_i (y_N - y_0)) / (x_N - x_0)
      q_i = y_{i-1} - α_i y_0 - p_i x_0

    Args:
        x: Nodos de interpolación en x (ordenados, sin NaN, únicos)
        y: Nodos de interpolación en y (mismo tamaño que x)
        alpha: Escalar o iterable con |alpha_i|<1 por intervalo. Controla la rugosidad:
               - α cercano a 0: curva más suave
               - α cercano a 1: curva más rugosa/fractal

    Returns:
        FIFParams con todos los coeficientes calculados

    Raises:
        ValueError: Si hay menos de 2 nodos, x no es creciente, o |α| >= 1
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    N = len(x) - 1

    if N < 1:
        raise ValueError("Se requieren al menos 2 nodos para la FIF.")

    if np.any(np.diff(x) <= 0):
        raise ValueError("Los x deben estar estrictamente crecientes.")

    x0, xN = x[0], x[-1]
    y0, yN = y[0], y[-1]

    if isinstance(alpha, (list, tuple, np.ndarray)):
        alpha_i = np.asarray(alpha, float)
        if alpha_i.size != N:
            raise ValueError(f"Si proporcionas lista de α, debe tener tamaño N={N}.")
    else:
        alpha_i = np.full(N, float(alpha))

    if np.any(np.abs(alpha_i) >= 1):
        raise ValueError("|α_i| debe ser < 1 para contracción vertical.")

    denom_x = (xN - x0)
    if denom_x == 0:
        raise ValueError("x0 == xN: nodos degenerados.")

    a = (x[1:] - x[:-1]) / denom_x
    b = (x[:-1] * xN - x[1:] * x0) / denom_x
    p = ((y[1:] - y[:-1]) - alpha_i * (yN - y0)) / denom_x
    q = y[:-1] - alpha_i * y0 - p * x0

    return FIFParams(
        a=a, b=b, alpha=alpha_i, p=p, q=q,
        x0=x0, xN=xN, y0=y0, yN=yN,
        x_nodes=x, y_nodes=y
    )


# ------------------------------------------
# Evaluación determinista de la FIF
# ------------------------------------------

def fif_evaluate(
    x_eval: np.ndarray,
    params: FIFParams,
    n_refinements: int = 12
) -> np.ndarray:
    """
    Evalúa la FIF en puntos x_eval usando la ecuación funcional por retro-iteración.

    Estrategia:
      Para cada x en [x_{i-1}, x_i], usamos la inversa del mapeo x' = a_i x + b_i:
        x_pre = (x - b_i)/a_i   y  f(x) = α_i f(x_pre) + p_i x_pre + q_i
      Repetimos hacia atrás hasta caer (casi) en {x0, xN}; inicializamos con
      interpolación lineal.

    Args:
        x_eval: Puntos donde evaluar la FIF
        params: Parámetros de la FIF (obtenidos con build_fif_params)
        n_refinements: Número de iteraciones de mejora (12 suele ser suficiente)

    Returns:
        Array con valores y evaluados en x_eval
    """
    x_eval = np.asarray(x_eval, float)
    y = np.interp(x_eval, params.x_nodes, params.y_nodes)  # inicialización lineal

    # Precomputo límites de intervalos
    xL = params.x_nodes[:-1]
    xR = params.x_nodes[1:]

    # Para ubicar qué intervalo contiene cada x
    def interval_index(x):
        # i tal que x in [x_{i-1}, x_i]; usamos búsqueda binaria
        i = np.searchsorted(params.x_nodes, x, side='right') - 1
        # Clamp
        return int(np.clip(i, 0, len(xL)-1))

    for _ in range(n_refinements):
        y_new = y.copy()
        for k, xk in enumerate(x_eval):
            i = interval_index(xk)
            a_i, b_i = params.a[i], params.b[i]
            alpha_i, p_i, q_i = params.alpha[i], params.p[i], params.q[i]

            # preimagen en el dominio "unidad"
            x_pre = (xk - b_i) / a_i

            # f(x) = α f(x_pre) + p x_pre + q
            # aproximamos f(x_pre) con la y actual (que ya está bastante cerca y se refina)
            if not (params.x0 <= x_pre <= params.xN):
                f_pre = np.interp(x_pre, params.x_nodes, params.y_nodes)
            else:
                f_pre = np.interp(x_pre, x_eval, y)

            y_new[k] = alpha_i * f_pre + p_i * x_pre + q_i
        y = y_new

    return y


# ------------------------------------------
# Reconstrucción de huecos
# ------------------------------------------

def fractal_interpolate_series(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float | Iterable[float] = 0.3,
    x_out: Optional[np.ndarray] = None,
    n_refinements: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye una FIF con los puntos conocidos (sin NaN) y evalúa en x_out.

    Esta es la función principal para interpolar series temporales con huecos
    usando interpolación fractal.

    Args:
        x: Coordenadas x originales (pueden contener NaN)
        y: Coordenadas y originales (pueden contener NaN en los huecos)
        alpha: Factor de rugosidad |α|<1. Valores típicos:
               - 0.0-0.3: interpolación suave
               - 0.3-0.6: rugosidad moderada
               - 0.6-0.9: altamente fractal
        x_out: Puntos donde evaluar. Si None, usa grilla uniforme con 5x puntos
        n_refinements: Iteraciones de refinamiento (12 es buen default)

    Returns:
        Tupla (x_out, y_hat) con las coordenadas interpoladas

    Raises:
        ValueError: Si no hay puntos conocidos suficientes
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # filtrar conocidos
    known_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(known_mask):
        raise ValueError("No hay puntos y conocidos (no NaN) para ajustar la FIF.")

    xk, yk = _clean_known_points(x[known_mask], y[known_mask])

    params = build_fif_params(xk, yk, alpha=alpha)

    if x_out is None:
        n = max(500, 5 * len(xk))
        x_out = np.linspace(params.x0, params.xN, n)
    else:
        x_out = np.asarray(x_out, float)

    y_out = fif_evaluate(x_out, params, n_refinements=n_refinements)

    return x_out, y_out


# ------------------------------------------
# Chaos game (opcional/ilustrativo)
# ------------------------------------------

def chaos_game_sample(
    params: FIFParams,
    n_points: int = 10000,
    burn_in: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera puntos (x,y) sobre la curva de la FIF usando iteración aleatoria de IFS.

    Útil para visualizar la estructura fractal; no reemplaza la evaluación
    determinista al interpolar.

    Args:
        params: Parámetros de la FIF
        n_points: Número de puntos a generar
        burn_in: Puntos iniciales a descartar
        seed: Semilla para reproducibilidad

    Returns:
        Tupla (xs, ys) con los puntos generados
    """
    rng = np.random.default_rng(seed)
    x = 0.5 * (params.x0 + params.xN)
    y = 0.5 * (params.y0 + params.yN)

    xs, ys = [], []
    for t in range(n_points + burn_in):
        i = rng.integers(0, len(params.a))
        x = params.a[i] * x + params.b[i]
        y = params.alpha[i] * y + params.p[i] * x + params.q[i]
        if t >= burn_in:
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)
