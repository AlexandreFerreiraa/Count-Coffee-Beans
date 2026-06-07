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

from typing import Dict, List


def effective_volume(V_internal_cm3: List[float], f_fill: List[float]) -> List[float]:
    """Volume efetivo (cm^3) elemento a elemento sobre amostras Monte Carlo."""
    return [V * f for V, f in zip(V_internal_cm3, f_fill)]


def effective_volume_from_params(p: Dict[str, List[float]]) -> List[float]:
    return effective_volume(p["V_internal_cm3"], p["f_fill"])
