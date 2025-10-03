#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dimensionalidad_utils.py
------------------------
Estimación de dimensión fractal por conteo de cajas (Minkowski–Bouligand)
a partir de una **imagen** (PNG/JPG/TIFF...).

Soporta:
- Umbralización simple (threshold) e inversión opcional.
- Selección automática de la mejor ventana log-log (max |R|).
- Escalas ε = (base)^(-k), p.ej. base=2 (por defecto) o base=3 (Cantor/Koch).

Basado en dimensionalidad.py original (manteniendo toda la funcionalidad).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None


# ---------------------------------------------------------------------
# Utilidades matemáticas y de ajuste
# ---------------------------------------------------------------------

@dataclass
class FitResult:
    """
    Resultado del ajuste lineal en espacio log-log.

    Attributes:
        slope: Pendiente de la recta (dimensión fractal)
        intercept: Intercepto de la recta
        rvalue: Coeficiente de correlación R
        used_x: Valores x usados en el ajuste
        used_y: Valores y usados en el ajuste
        idx_used: Índices de los puntos usados
    """
    slope: float
    intercept: float
    rvalue: float
    used_x: np.ndarray
    used_y: np.ndarray
    idx_used: np.ndarray


def _linear_fit(x: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Ajuste lineal y = a*x + b, devolviendo pendiente, intercepto y R.

    Args:
        x: Coordenadas x (debe ser 1D)
        y: Coordenadas y (debe ser 1D, mismo tamaño que x)

    Returns:
        FitResult con los parámetros del ajuste

    Raises:
        ValueError: Si x e y no son 1D o no tienen el mismo tamaño
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x e y deben ser vectores 1D del mismo tamaño.")

    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    denom = np.dot(dx, dx)

    if denom == 0:
        raise ValueError("Varianza nula en x; no se puede ajustar.")

    slope = float(np.dot(dx, dy) / denom)
    intercept = float(ym - slope * xm)
    rden = math.sqrt(np.dot(dx, dx) * np.dot(dy, dy)) if np.dot(dy, dy) > 0 else 0.0
    rvalue = float(np.dot(dx, dy) / rden) if rden > 0 else 0.0

    return FitResult(slope, intercept, rvalue, x, y, np.arange(len(x)))


def _auto_scale_window(x: np.ndarray, y: np.ndarray, min_points: int = 5) -> FitResult:
    """
    Selecciona automáticamente la subventana contigua con mayor |R| usando
    al menos 'min_points' puntos, para evitar escalas contaminadas.

    Args:
        x: Coordenadas x en espacio log
        y: Coordenadas y en espacio log
        min_points: Número mínimo de puntos para considerar válida una ventana

    Returns:
        FitResult de la mejor ventana encontrada
    """
    best: Optional[FitResult] = None
    n = len(x)

    for i in range(n):
        for j in range(i + min_points, n + 1):
            fit = _linear_fit(x[i:j], y[i:j])
            if best is None or abs(fit.rvalue) > abs(best.rvalue):
                fit.idx_used = np.arange(i, j)
                best = fit

    return best


# ---------------------------------------------------------------------
# Box-counting para imagen (convierte a nube de puntos y cuenta celdas)
# ---------------------------------------------------------------------

def box_count(points: np.ndarray, eps: float) -> int:
    """
    Cuenta celdas de una grilla de tamaño 'eps' que contienen ≥1 punto.

    Args:
        points: Array de puntos normalizado a [0,1]^d
        eps: Tamaño de la celda (debe estar en (0, 1])

    Returns:
        Número de celdas ocupadas

    Raises:
        ValueError: Si eps no está en el rango válido
    """
    if not (0 < eps <= 1.0):
        raise ValueError("eps debe estar en (0, 1].")

    if points.size == 0:
        return 0

    g = np.floor(points / eps).astype(np.int64)
    if g.ndim == 1:
        g = g.reshape(-1, 1)

    cells_bytes = np.ascontiguousarray(g).view(
        np.dtype((np.void, g.dtype.itemsize * g.shape[1]))
    )

    return np.unique(cells_bytes).size


def box_counting_dimension(
    points: np.ndarray,
    eps_list: Optional[Iterable[float]] = None,
    auto_window: bool = True,
    min_points_fit: int = 5,
    return_series: bool = False
) -> Tuple[float, FitResult, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Estima la dimensión fractal por conteo de cajas usando N(ε) ~ (1/ε)^D.

    Args:
        points: Array de puntos (nx2 o nx3) que forman el conjunto fractal
        eps_list: Lista de tamaños de celda (escalas). Si None, usa 2^(-k) para k=2..9
        auto_window: Si True, selecciona automáticamente la mejor ventana lineal
        min_points_fit: Número mínimo de escalas para considerar en el ajuste
        return_series: Si True, devuelve también los datos log-log (xs, ys)

    Returns:
        Tupla (D, fit, series) donde:
            - D: Dimensión fractal estimada (pendiente del ajuste)
            - fit: FitResult con información del ajuste lineal
            - series: (xs, ys) datos log-log si return_series=True, None si no

    Raises:
        ValueError: Si hay menos de 2 puntos o no hay escalas informativas
    """
    P = np.asarray(points, float)

    if P.ndim == 1:
        P = P.reshape(-1, 1)

    if P.shape[0] < 2:
        raise ValueError("Se requieren al menos 2 puntos.")

    # Normaliza a [0,1]^d para quitar escala/traslación
    mn, mx = np.nanmin(P, axis=0), np.nanmax(P, axis=0)
    span = mx - mn
    span[span == 0] = 1.0
    P = (P - mn) / span

    if eps_list is None:
        eps_list = [2.0 ** (-k) for k in range(2, 10)]  # 1/4..1/512 por defecto

    eps = np.array(list(eps_list), dtype=float)
    counts = np.array([box_count(P, e) for e in eps], float)

    mask = counts > 1  # descarta escalas sin información
    eps, counts = eps[mask], counts[mask]

    if len(eps) < min_points_fit:
        raise ValueError("No hay suficientes escalas informativas (>1 celda).")

    xs = np.log(1.0 / eps)   # log(1/ε)
    ys = np.log(counts)      # log N(ε)

    fit = _auto_scale_window(xs, ys, min_points=min_points_fit) if auto_window else _linear_fit(xs, ys)
    D = max(0.0, float(fit.slope))
    series = (xs, ys) if return_series else None

    return D, fit, series


# ---------------------------------------------------------------------
# Lectura de imagen -> puntos
# ---------------------------------------------------------------------

def load_points_from_image(
    path: str,
    threshold: float = 0.5,
    invert: bool = False
) -> np.ndarray:
    """
    Carga imagen y devuelve coordenadas (x,y) de píxeles 'activos' como puntos.

    Args:
        path: Ruta al archivo de imagen
        threshold: Umbral de binarización en [0,1]. Píxeles con intensidad >= threshold son activos
        invert: Si True, invierte la máscara (útil cuando el fondo es claro)

    Returns:
        Array nx2 con coordenadas (x,y) de los píxeles activos

    Raises:
        ImportError: Si Pillow no está instalado
    """
    if Image is None:
        raise ImportError("Pillow no está instalado. Instala con: pip install pillow")

    img = Image.open(path).convert("L")  # escala de grises
    arr = np.asarray(img, dtype=float) / 255.0
    mask = arr >= threshold

    if invert:
        mask = ~mask

    ys, xs = np.nonzero(mask)  # filas, columnas
    pts = np.stack([xs, ys], axis=1).astype(float)

    return pts
