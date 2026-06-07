"""
packing — modelos de fração de empacotamento (random packing).

A fração de empacotamento phi = (volume de sólido dos grãos) / (volume ocupado).
O complemento (1 - phi) é o ar entre os grãos.

Dois caminhos:

1) phi como PRIOR direto (Modelo A): phi ~ 0.55-0.65 para grãos irregulares
   alongados (random packing com atrito), informado pela densidade a granel.

2) phi A PARTIR DA FORMA (Modelo D): para esferoides, a densidade de
   empacotamento aleatório (random close packing) cresce de ~0.64 (esfera,
   razão de aspecto a=1) até um pico ~0.70-0.71 perto de a~1.3-1.5 e decresce
   para razões maiores (Donev et al., Science 2004 — valores reimplementados).
   Grãos reais têm atrito/irregularidade => multiplicamos por um fator < 1.
"""

from __future__ import annotations

from typing import List


def phi_spheroid(aspect: float) -> float:
    """Densidade de random packing de um esferoide liso em função da razão
    de aspecto `aspect` (>= 1). Curva suave e contínua calibrada para reproduzir
    o comportamento qualitativo de Donev et al. (Science, 2004):
        a=1.0 -> ~0.64 (esferas, random close packing)
        pico  -> ~0.712 perto de a~1.4
        a=2.0 -> ~0.705 ; a=3.0 -> ~0.66 (decaimento suave; esferoides prolatos
                 mantêm densidade alta numa faixa ampla de aspecto)
    Piso físico em 0.58.
    """
    a = max(1.0, aspect)
    peak = 0.712
    a_peak = 1.4
    if a <= a_peak:
        # interpola suavemente entre 0.64 (esfera) e o pico
        t = (a - 1.0) / (a_peak - 1.0)
        return 0.64 + t * (peak - 0.64)
    # decaimento parabólico brando após o pico (coef. pequeno => faixa ampla)
    phi = peak - 0.020 * (a - a_peak) ** 2
    return max(phi, 0.58)


def phi_from_shape(aspect: List[float], fric_factor: List[float]) -> List[float]:
    """phi efetivo de grãos reais = phi_esferoide(aspecto) * fator de atrito."""
    return [phi_spheroid(a) * f for a, f in zip(aspect, fric_factor)]
