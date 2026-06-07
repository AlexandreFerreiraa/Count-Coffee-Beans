"""
geometry — volume interno efetivo ocupado pela coluna de grãos.

Modelamos a região ocupada pelos grãos como um cilindro (corpo do pote) de
diâmetro interno `d_in` e altura `h_beans` (altura da coluna de grãos), com uma
correção de forma `k_shape` (fundo arredondado, irregularidades).

Usar a ALTURA DA COLUNA de grãos (em vez da altura total do pote) embute
diretamente o espaço vazio do pescoço — coerente com a observação de que os
grãos não chegam ao topo.

    V_eff = pi * (d_in/2)^2 * h_beans * k_shape       [cm^3]
"""

from __future__ import annotations

from math import pi
from typing import Dict, List


def effective_volume(d_in_cm: List[float], h_beans_cm: List[float], k_shape: List[float]) -> List[float]:
    """Volume efetivo (cm^3) elemento a elemento sobre amostras Monte Carlo."""
    return [pi * (d / 2.0) ** 2 * h * k for d, h, k in zip(d_in_cm, h_beans_cm, k_shape)]


def effective_volume_from_params(p: Dict[str, List[float]]) -> List[float]:
    return effective_volume(p["d_in_cm"], p["h_beans_cm"], p["k_shape"])
