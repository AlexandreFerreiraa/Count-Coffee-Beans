"""
mcstats — utilitários de Monte Carlo e estatística usando apenas a stdlib.

Tudo aqui opera sobre listas de floats. Sem numpy/scipy, para o núcleo rodar em
qualquer máquina sem instalação. As implementações seguem definições padrão:

- Amostragem de normal truncada, lognormal (parametrizada por média/desvio).
- KDE gaussiano (largura de banda de Silverman) para densidades suaves.
- Coeficiente de Bhattacharyya e divergência de Jensen-Shannon (sobre grade comum).
- Teste de Kolmogorov-Smirnov de duas amostras (estatística D + p-valor assintótico).
- Bootstrap para erro de Monte Carlo (MCSE) e IC de estatísticas.
- Índices de Sobol de 1ª ordem e totais (estimador de Saltelli).
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------
class RNG:
    """Gerador encapsulado para reprodutibilidade e amostragem de specs."""

    def __init__(self, seed: int = 0):
        self._r = random.Random(seed)

    def normal(self, mean: float, sd: float) -> float:
        return self._r.gauss(mean, sd)

    def uniform(self, a: float, b: float) -> float:
        return self._r.uniform(a, b)

    def random(self) -> float:
        return self._r.random()

    def spawn(self, offset: int) -> "RNG":
        """Sub-gerador determinístico (para colunas independentes em Sobol)."""
        return RNG(self._r.randint(0, 2**31 - 1) + offset)


def _truncnorm_one(rng: RNG, mean: float, sd: float, lo: float, hi: float) -> float:
    """Amostra normal truncada por rejeição, com fallback de clamp."""
    for _ in range(100):
        x = rng.normal(mean, sd)
        if lo <= x <= hi:
            return x
    # fallback: clamp (raríssimo se truncamento não for extremo)
    return min(max(rng.normal(mean, sd), lo), hi)


def sample_spec(spec: dict, n: int, rng: RNG) -> List[float]:
    """Amostra n valores de um *spec* de parâmetro (ver config.PARAMS)."""
    dist = spec["dist"]
    if dist == "const":
        v = spec["value"]
        return [v] * n
    if dist == "normal":
        m, s = spec["mean"], spec["sd"]
        return [rng.normal(m, s) for _ in range(n)]
    if dist == "truncnormal":
        m, s, lo, hi = spec["mean"], spec["sd"], spec["lo"], spec["hi"]
        return [_truncnorm_one(rng, m, s, lo, hi) for _ in range(n)]
    if dist == "lognormal":
        # parametrizado por média/desvio do espaço LINEAR
        m, s = spec["mean"], spec["sd"]
        sigma2 = math.log(1.0 + (s * s) / (m * m))
        mu = math.log(m) - 0.5 * sigma2
        sigma = math.sqrt(sigma2)
        return [math.exp(rng.normal(mu, sigma)) for _ in range(n)]
    raise ValueError(f"Distribuição desconhecida: {dist}")


# ---------------------------------------------------------------------------
# Estatística descritiva
# ---------------------------------------------------------------------------
def mean(x: Sequence[float]) -> float:
    return sum(x) / len(x)


def variance(x: Sequence[float], ddof: int = 1) -> float:
    n = len(x)
    if n - ddof <= 0:
        return 0.0
    m = mean(x)
    return sum((xi - m) ** 2 for xi in x) / (n - ddof)


def stdev(x: Sequence[float], ddof: int = 1) -> float:
    return math.sqrt(variance(x, ddof))


def percentile(sorted_x: Sequence[float], q: float) -> float:
    """Percentil (q em [0,100]) por interpolação linear. Requer entrada ORDENADA."""
    if not sorted_x:
        return float("nan")
    if q <= 0:
        return sorted_x[0]
    if q >= 100:
        return sorted_x[-1]
    pos = (len(sorted_x) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_x[int(pos)]
    frac = pos - lo
    return sorted_x[lo] * (1 - frac) + sorted_x[hi] * frac


def median(x: Sequence[float]) -> float:
    return percentile(sorted(x), 50.0)


def cov(x: Sequence[float]) -> float:
    """Coeficiente de variação (desvio/média) — métrica de precisão relativa."""
    m = mean(x)
    return stdev(x) / m if m != 0 else float("inf")


def summary(x: Sequence[float]) -> Dict[str, float]:
    s = sorted(x)
    return {
        "mean": mean(s),
        "median": percentile(s, 50.0),
        "sd": stdev(s),
        "cov": cov(s),
        "p2.5": percentile(s, 2.5),
        "p5": percentile(s, 5.0),
        "p25": percentile(s, 25.0),
        "p75": percentile(s, 75.0),
        "p95": percentile(s, 95.0),
        "p97.5": percentile(s, 97.5),
        "min": s[0],
        "max": s[-1],
        "n": len(s),
    }


# ---------------------------------------------------------------------------
# Densidades (KDE) e comparação de distribuições
# ---------------------------------------------------------------------------
def silverman_bandwidth(x: Sequence[float]) -> float:
    n = len(x)
    s = stdev(x)
    sx = sorted(x)
    iqr = percentile(sx, 75) - percentile(sx, 25)
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    if sigma <= 0:
        sigma = abs(mean(x)) * 1e-3 + 1e-9
    return 0.9 * sigma * n ** (-1.0 / 5.0)


def kde_on_grid(x: Sequence[float], grid: Sequence[float], bw: float | None = None) -> List[float]:
    """Densidade KDE gaussiana avaliada na grade. O(len(grid)*len(x)).

    Para eficiência com amostras grandes, subamostramos para até 4000 pontos.
    """
    if bw is None:
        bw = silverman_bandwidth(x)
    xs = x
    if len(x) > 4000:
        step = len(x) // 4000
        xs = x[::step]
    n = len(xs)
    inv = 1.0 / (bw * math.sqrt(2 * math.pi))
    two_bw2 = 2 * bw * bw
    out = []
    for g in grid:
        acc = 0.0
        for xi in xs:
            d = g - xi
            acc += math.exp(-(d * d) / two_bw2)
        out.append(inv * acc / n)
    return out


def common_grid(a: Sequence[float], b: Sequence[float], m: int = 256) -> List[float]:
    lo = min(min(a), min(b))
    hi = max(max(a), max(b))
    pad = 0.05 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad
    step = (hi - lo) / (m - 1)
    return [lo + i * step for i in range(m)]


def _normalize_density(p: List[float], dx: float) -> List[float]:
    z = sum(p) * dx
    if z <= 0:
        return p
    return [pi_ / z for pi_ in p]


def bhattacharyya_coefficient(a: Sequence[float], b: Sequence[float]) -> float:
    """BC em [0,1]; 1 = distribuições idênticas. Mede sobreposição."""
    grid = common_grid(a, b)
    dx = grid[1] - grid[0]
    pa = _normalize_density(kde_on_grid(a, grid), dx)
    pb = _normalize_density(kde_on_grid(b, grid), dx)
    return sum(math.sqrt(pi_ * qi) for pi_, qi in zip(pa, pb)) * dx


def overlap_coefficient(a: Sequence[float], b: Sequence[float]) -> float:
    """Área de sobreposição (integral do mínimo das densidades), em [0,1]."""
    grid = common_grid(a, b)
    dx = grid[1] - grid[0]
    pa = _normalize_density(kde_on_grid(a, grid), dx)
    pb = _normalize_density(kde_on_grid(b, grid), dx)
    return sum(min(pi_, qi) for pi_, qi in zip(pa, pb)) * dx


def jensen_shannon_divergence(a: Sequence[float], b: Sequence[float]) -> float:
    """JSD em bits (base 2), em [0,1]. 0 = idênticas, 1 = disjuntas."""
    grid = common_grid(a, b)
    dx = grid[1] - grid[0]
    pa = _normalize_density(kde_on_grid(a, grid), dx)
    pb = _normalize_density(kde_on_grid(b, grid), dx)

    def kl(p, q):
        s = 0.0
        for pi_, qi in zip(p, q):
            if pi_ > 0 and qi > 0:
                s += pi_ * math.log2(pi_ / qi) * dx
        return s

    m = [(pi_ + qi) / 2.0 for pi_, qi in zip(pa, pb)]
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def ks_2samp(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Kolmogorov-Smirnov de 2 amostras. Retorna (D, p-valor assintótico)."""
    xa = sorted(a)
    xb = sorted(b)
    na, nb = len(xa), len(xb)
    ia = ib = 0
    cdf_a = cdf_b = 0.0
    d = 0.0
    while ia < na and ib < nb:
        if xa[ia] <= xb[ib]:
            v = xa[ia]
            while ia < na and xa[ia] == v:
                ia += 1
            cdf_a = ia / na
        else:
            v = xb[ib]
            while ib < nb and xb[ib] == v:
                ib += 1
            cdf_b = ib / nb
        d = max(d, abs(cdf_a - cdf_b))
    en = math.sqrt(na * nb / (na + nb))
    # p-valor pela distribuição assintótica de Kolmogorov
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2 * lam * lam * k * k) for k in range(1, 101))
    p = max(0.0, min(1.0, p))
    return d, p


# ---------------------------------------------------------------------------
# Bootstrap (MCSE / IC de estatísticas)
# ---------------------------------------------------------------------------
def bootstrap_statistic(
    x: Sequence[float], stat: Callable[[Sequence[float]], float], n_boot: int, rng: RNG
) -> Dict[str, float]:
    n = len(x)
    vals = []
    for _ in range(n_boot):
        sample = [x[int(rng.random() * n)] for _ in range(n)]
        vals.append(stat(sample))
    vs = sorted(vals)
    return {
        "estimate": stat(x),
        "se": stdev(vs),
        "ci_lo": percentile(vs, 2.5),
        "ci_hi": percentile(vs, 97.5),
    }


def running_mean(x: Sequence[float], n_points: int = 200) -> Tuple[List[int], List[float]]:
    """Média acumulada amostrada em n_points (diagnóstico de convergência)."""
    n = len(x)
    idxs = sorted(set(int(round(i)) for i in _linspace(1, n, n_points)))
    out_idx, out_val = [], []
    acc = 0.0
    j = 0
    for i in range(1, n + 1):
        acc += x[i - 1]
        if j < len(idxs) and i == idxs[j]:
            out_idx.append(i)
            out_val.append(acc / i)
            j += 1
    return out_idx, out_val


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Sensibilidade de Sobol (estimador de Saltelli)
# ---------------------------------------------------------------------------
def sobol_indices(
    compute: Callable[[Dict[str, List[float]]], List[float]],
    specs: Dict[str, dict],
    n: int,
    rng: RNG,
) -> Dict[str, Dict[str, float]]:
    """Índices de Sobol de 1ª ordem (S1) e total (ST) por parâmetro.

    compute: função que recebe dict {param: lista} e devolve lista de saídas.
    specs:   dict {param: spec} dos parâmetros do modelo.

    Estimadores (Saltelli 2010):
        S1_i = (1/N) Σ f(B)·(f(A_B^i) − f(A)) / Var(Y)
        ST_i = (1/2N) Σ (f(A) − f(A_B^i))² / Var(Y)
    """
    names = list(specs.keys())
    # duas matrizes de amostras independentes
    A = {nm: sample_spec(specs[nm], n, rng.spawn(i)) for i, nm in enumerate(names)}
    B = {nm: sample_spec(specs[nm], n, rng.spawn(100 + i)) for i, nm in enumerate(names)}

    fA = compute(A)
    fB = compute(B)
    all_y = fA + fB
    varY = variance(all_y)
    if varY <= 0:
        return {nm: {"S1": 0.0, "ST": 0.0} for nm in names}

    out: Dict[str, Dict[str, float]] = {}
    for nm in names:
        AB = dict(A)
        AB[nm] = B[nm]  # substitui a coluna nm por B
        fAB = compute(AB)
        s1 = sum(fB[j] * (fAB[j] - fA[j]) for j in range(n)) / n / varY
        st = sum((fA[j] - fAB[j]) ** 2 for j in range(n)) / (2 * n) / varY
        out[nm] = {"S1": max(0.0, s1), "ST": max(0.0, st)}
    return out
