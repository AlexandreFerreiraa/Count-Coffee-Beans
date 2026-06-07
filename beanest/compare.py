"""
compare — comparação estatística entre modelos, sensibilidade e convergência.

Como NÃO há "verdade-terreno" (não contamos os grãos reais), não é possível
medir acurácia diretamente. Em vez disso avaliamos:

1) CONCORDÂNCIA entre métodos (modelos semi-independentes que convergem reforçam
   a credibilidade do resultado):
   - Coeficiente de sobreposição e de Bhattacharyya (quanto as distribuições se
     sobrepõem);
   - Divergência de Jensen-Shannon (0 = idênticas, 1 = disjuntas);
   - Teste KS de 2 amostras (D e p-valor) — testa se duas distribuições diferem.

2) PRECISÃO de cada modelo: coeficiente de variação (CoV) e largura do IC95%.

3) SENSIBILIDADE (Sobol S1/ST): quais parâmetros dominam a variância de cada
   modelo. Identifica onde investir esforço de medição para reduzir incerteza.

4) CONVERGÊNCIA de Monte Carlo: erro-padrão de Monte Carlo (MCSE) da mediana do
   ensemble por bootstrap; média acumulada (estabilização).
"""

from __future__ import annotations

from typing import Dict, List

from . import mcstats
from .config import N_SOBOL, PARAMS, EXP_MASS_PER_BEAN_G
from .mcstats import RNG


def descriptive_table(samples: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    return {k: mcstats.summary(v) for k, v in samples.items()}


def _subsample(x: List[float], k: int, rng: RNG) -> List[float]:
    if len(x) <= k:
        return list(x)
    n = len(x)
    return [x[int(rng.random() * n)] for _ in range(k)]


def pairwise_agreement(samples: Dict[str, List[float]], ks_n: int = 2000, seed: int = 5) -> List[Dict]:
    """Métricas par-a-par de concordância entre modelos.

    O KS é calculado sobre uma SUBAMOSTRA (ks_n) porque, com dezenas de milhares
    de amostras de Monte Carlo, qualquer diferença ínfima vira "significativa"
    (p->0) — o p-valor deixaria de ser informativo. Overlap, Bhattacharyya e a
    divergência de Jensen-Shannon são as métricas de concordância confiáveis aqui.
    """
    rng = RNG(seed)
    keys = list(samples.keys())
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = samples[keys[i]], samples[keys[j]]
            d, p = mcstats.ks_2samp(_subsample(a, ks_n, rng), _subsample(b, ks_n, rng))
            rows.append({
                "model_a": keys[i],
                "model_b": keys[j],
                "overlap": mcstats.overlap_coefficient(a, b),
                "bhattacharyya": mcstats.bhattacharyya_coefficient(a, b),
                "js_divergence": mcstats.jensen_shannon_divergence(a, b),
                "ks_D": d,
                "ks_p": p,
            })
    return rows


def sensitivity(models: Dict, n: int = N_SOBOL, seed: int = 7) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Índices de Sobol (S1, ST) por parâmetro, para cada modelo base."""
    rng = RNG(seed)
    out = {}
    for i, (key, model) in enumerate(models.items()):
        out[key] = mcstats.sobol_indices(model.compute, model.param_specs(), n, rng.spawn(i))
    return out


def convergence(ensemble_samples: List[float], n_boot: int = 800, seed: int = 11) -> Dict:
    """MCSE da mediana (bootstrap) + traço da média acumulada."""
    rng = RNG(seed)
    boot = mcstats.bootstrap_statistic(ensemble_samples, mcstats.median, n_boot, rng)
    idx, run = mcstats.running_mean(ensemble_samples, n_points=200)
    return {
        "median_estimate": boot["estimate"],
        "median_mcse": boot["se"],
        "median_ci_lo": boot["ci_lo"],
        "median_ci_hi": boot["ci_hi"],
        "running_idx": idx,
        "running_mean": run,
    }


def cross_consistency_with_experiment(samples: Dict[str, List[float]]) -> Dict[str, float]:
    """Verificação cruzada leve: a massa total implícita por cada modelo é
    fisicamente plausível? (Sanidade, não validação.) Retorna a massa total
    mediana implícita (kg) usando a massa/grão consistente com o experimento
    (m_exp * E[s_lin^3])."""
    s = PARAMS["s_lin"]
    # E[s^3] para normal truncada ~ aproxima por mean^3 + 3*mean*sd^2 (correção de Jensen)
    e_s3 = s["mean"] ** 3 + 3 * s["mean"] * (s["sd"] ** 2)
    g_per_bean = EXP_MASS_PER_BEAN_G * e_s3
    out = {}
    for k, v in samples.items():
        out[k] = mcstats.median(v) * g_per_bean / 1000.0
    return out
