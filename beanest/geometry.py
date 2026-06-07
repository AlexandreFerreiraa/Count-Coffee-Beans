"""
geometry — volume interno efetivo ocupado pela coluna de grãos.

Com a CAPACIDADE INTERNA medida do pote disponível (3223 cm^3), o volume
ocupado pelos grãos é simplesmente a capacidade vezes a fração preenchida:

    V_eff = V_internal * f_fill          [cm^3]

onde f_fill (< 1) desconta o headspace vazio do pescoço (os grãos não chegam ao
topo). Esta abordagem é mais fiel que aproximar o pote por um cilindro a partir
de diâmetro/altura, pois a capacidade medida já incorpora o afunilamento do
pescoço — e reduz drasticamente a incerteza geométrica (que antes dominava).
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Dict, List


def effective_volume(V_internal_cm3: List[float], f_fill: List[float]) -> List[float]:
    """Volume efetivo (cm^3) elemento a elemento sobre amostras Monte Carlo."""
    return [V * f for V, f in zip(V_internal_cm3, f_fill)]


def effective_volume_from_params(p: Dict[str, List[float]]) -> List[float]:
    return effective_volume(p["V_internal_cm3"], p["f_fill"])


# ---------------------------------------------------------------------------
# Diâmetro interno e coerência geométrica (NÃO afeta a contagem de N — que usa o
# volume medido — mas valida o conjunto de medidas e calibra a visão).
# ---------------------------------------------------------------------------
def internal_diameter_cm(d_ext_cm: float, wall_cm: float) -> float:
    """Diâmetro interno = externo - 2 * espessura da parede."""
    return d_ext_cm - 2.0 * wall_cm


def geometry_consistency(
    d_ext_cm: float,
    height_cm: float,
    V_internal_cm3: float,
    wall_lo_cm: float,
    wall_hi_cm: float,
) -> Dict[str, float]:
    """Checa se (diâmetro externo, altura, volume) são coerentes.

    - D_in_lo/hi: diâmetro interno do corpo a partir da faixa de espessura.
    - d_eq_full: diâmetro de um cilindro de altura `height_cm` que teria
      exatamente o volume medido (subestima o corpo, pois o pescoço afunila).
    - h_eff_body: altura útil de um cilindro com o D_in central que conteria o
      volume medido (deve ser < altura total, pela presença do pescoço).
    """
    d_in_hi = internal_diameter_cm(d_ext_cm, wall_lo_cm)  # parede fina -> interno maior
    d_in_lo = internal_diameter_cm(d_ext_cm, wall_hi_cm)
    d_in_mid = 0.5 * (d_in_lo + d_in_hi)

    d_eq_full = 2.0 * sqrt(V_internal_cm3 / (pi * height_cm))
    h_eff_body = V_internal_cm3 / (pi * (d_in_mid / 2.0) ** 2)
    return {
        "d_ext_cm": d_ext_cm,
        "d_in_lo_cm": d_in_lo,
        "d_in_hi_cm": d_in_hi,
        "d_in_mid_cm": d_in_mid,
        "d_eq_full_cm": d_eq_full,
        "h_eff_body_cm": h_eff_body,
        "height_cm": height_cm,
        "V_internal_cm3": V_internal_cm3,
        # coerente se a altura útil do corpo for menor que a altura total
        # (sobra para o pescoço) e positiva:
        "coherent": 0 < h_eff_body < height_cm,
    }

