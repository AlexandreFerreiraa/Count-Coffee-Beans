"""
beans — propriedades de um grão individual (massa, volume, dimensões).

Duas rotas independentes para o volume/massa do grão:

1) Rota EXPERIMENTAL (ancorada no dado 89 grãos = 11 g):
       m_bean = m_exp * s_lin^3
   onde s_lin é o fator de escala linear dos grãos do pote vs. experimento
   (massa escala com o cubo do tamanho linear). O volume "envelope" sai de
       v_bean = m_bean / rho_app.

2) Rota GEOMÉTRICA (independente do experimento), usada no Modelo D:
   modela o grão como uma fração `kappa` do elipsoide envolvente de eixos
   L, W, T (mm):
       v_geo = kappa * (pi/6) * L * W * T   [mm^3]  -> /1000 -> cm^3
   O grão de café é ~meio-elipsoide "cheio" (uma face plana), daí kappa < 1.
"""

from __future__ import annotations

from math import pi
from typing import Dict, List

from .config import EXP_MASS_PER_BEAN_G


def bean_mass_experimental(s_lin: List[float]) -> List[float]:
    """Massa por grão (g) ancorada no experimento, escalada por s_lin^3."""
    return [EXP_MASS_PER_BEAN_G * (s ** 3) for s in s_lin]


def bean_volume_from_mass(mass_g: List[float], rho_app: List[float]) -> List[float]:
    """Volume envelope do grão (cm^3) = massa / densidade aparente."""
    return [m / r for m, r in zip(mass_g, rho_app)]


def bean_volume_geometric(
    L_mm: List[float], W_mm: List[float], T_mm: List[float], kappa: List[float]
) -> List[float]:
    """Volume envelope do grão (cm^3) pela rota geométrica (eixos + fator de forma)."""
    f = (pi / 6.0) / 1000.0  # mm^3 -> cm^3
    return [k * f * L * W * T for L, W, T, k in zip(L_mm, W_mm, T_mm, kappa)]


def aspect_ratio(L_mm: List[float], W_mm: List[float], T_mm: List[float]) -> List[float]:
    """Razão de aspecto = eixo maior / média dos dois menores (elongação)."""
    return [L / ((W + T) / 2.0) for L, W, T in zip(L_mm, W_mm, T_mm)]


# Conveniências orientadas a params (para Sobol / modelos)
def mass_from_params(p: Dict[str, List[float]]) -> List[float]:
    return bean_mass_experimental(p["s_lin"])


def volume_exp_from_params(p: Dict[str, List[float]]) -> List[float]:
    return bean_volume_from_mass(mass_from_params(p), p["rho_app"])
