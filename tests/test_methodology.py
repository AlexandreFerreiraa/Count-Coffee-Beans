"""
Testes de validação da metodologia (rodar: python -m pytest tests/ -q  OU  python tests/test_methodology.py).

Não dependem de libs externas. Validam que as ferramentas estatísticas estão
corretas em casos com resposta conhecida — credibilidade do pipeline.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beanest import mcstats, models
from beanest.mcstats import RNG


def test_sobol_ishigami():
    """Função de Ishigami: índices de Sobol de 1ª ordem ANALÍTICOS conhecidos.

    f = sin(x1) + a sin^2(x2) + b x3^4 sin(x1), xi ~ U(-pi, pi), a=7, b=0.1.
    Var = a^2/8 + b pi^4/5 + b^2 pi^8/18 + 1/2
    V1 = 1/2 (1 + b pi^4/5)^2 ; V2 = a^2/8 ; V3 = 0
    """
    a, b = 7.0, 0.1
    pi = math.pi
    # specs uniformes em [-pi, pi] via "truncnormal" largo? Não: criamos sampler uniforme.
    specs = {
        "x1": {"dist": "uniform", "lo": -pi, "hi": pi},
        "x2": {"dist": "uniform", "lo": -pi, "hi": pi},
        "x3": {"dist": "uniform", "lo": -pi, "hi": pi},
    }

    # mcstats.sample_spec não tem "uniform"; injetamos um sampler local
    def sample_uniform(spec, n, rng):
        return [rng.uniform(spec["lo"], spec["hi"]) for _ in range(n)]

    orig = mcstats.sample_spec
    mcstats.sample_spec = lambda spec, n, rng: (
        sample_uniform(spec, n, rng) if spec.get("dist") == "uniform" else orig(spec, n, rng)
    )
    try:
        def f(p):
            return [math.sin(x1) + a * math.sin(x2) ** 2 + b * (x3 ** 4) * math.sin(x1)
                    for x1, x2, x3 in zip(p["x1"], p["x2"], p["x3"])]

        idx = mcstats.sobol_indices(f, specs, n=20000, rng=RNG(1))
    finally:
        mcstats.sample_spec = orig

    var = a ** 2 / 8 + b * pi ** 4 / 5 + b ** 2 * pi ** 8 / 18 + 0.5
    s1_true = 0.5 * (1 + b * pi ** 4 / 5) ** 2 / var
    s2_true = (a ** 2 / 8) / var
    s3_true = 0.0

    assert abs(idx["x1"]["S1"] - s1_true) < 0.05, (idx["x1"]["S1"], s1_true)
    assert abs(idx["x2"]["S1"] - s2_true) < 0.05, (idx["x2"]["S1"], s2_true)
    assert abs(idx["x3"]["S1"] - s3_true) < 0.05, (idx["x3"]["S1"], s3_true)
    # x3 tem efeito de interação (ST3 > 0) embora S3 ~ 0
    assert idx["x3"]["ST"] > 0.05
    print("test_sobol_ishigami OK:",
          f"S1=({idx['x1']['S1']:.3f},{idx['x2']['S1']:.3f},{idx['x3']['S1']:.3f}) "
          f"esperado=({s1_true:.3f},{s2_true:.3f},{s3_true:.3f})")


def test_percentile_known():
    x = list(range(1, 101))  # 1..100
    assert abs(mcstats.percentile(x, 50) - 50.5) < 1e-9
    assert abs(mcstats.percentile(x, 0) - 1) < 1e-9
    assert abs(mcstats.percentile(x, 100) - 100) < 1e-9
    print("test_percentile_known OK")


def test_truncnorm_bounds():
    rng = RNG(3)
    spec = {"dist": "truncnormal", "mean": 0.0, "sd": 1.0, "lo": -1.0, "hi": 1.0}
    xs = mcstats.sample_spec(spec, 5000, rng)
    assert all(-1.0 <= x <= 1.0 for x in xs)
    print("test_truncnorm_bounds OK")


def test_ks_identical_vs_shifted():
    rng = RNG(4)
    a = [rng.normal(0, 1) for _ in range(3000)]
    b = [rng.normal(0, 1) for _ in range(3000)]
    c = [rng.normal(3, 1) for _ in range(3000)]
    d_ab, p_ab = mcstats.ks_2samp(a, b)
    d_ac, p_ac = mcstats.ks_2samp(a, c)
    assert p_ab > 0.01 and d_ab < 0.1      # iguais: não rejeita
    assert p_ac < 1e-6 and d_ac > 0.7      # deslocadas: rejeita forte
    print(f"test_ks OK: p_ab={p_ab:.3f} p_ac={p_ac:.2e}")


def test_pipeline_runs_and_is_plausible():
    res = models.run_all(n=8000, seed=42)
    s = mcstats.summary(res["ensemble_linear"])
    # ordem de grandeza plausível para um garrafão de ~3-4 L de café verde
    assert 5000 < s["median"] < 30000, s["median"]
    assert s["p2.5"] < s["median"] < s["p97.5"]
    # pesos por princípio somam 1 e A/B compartilham peso
    assert abs(sum(res["weights"].values()) - 1.0) < 1e-9
    assert res["model_weights"]["A_volumetrico"] == res["model_weights"]["B_massa_bulk"]
    print(f"test_pipeline OK: mediana={s['median']:.0f} IC95=[{s['p2.5']:.0f},{s['p97.5']:.0f}]")


if __name__ == "__main__":
    test_percentile_known()
    test_truncnorm_bounds()
    test_ks_identical_vs_shifted()
    test_sobol_ishigami()
    test_pipeline_runs_and_is_plausible()
    print("\nTodos os testes passaram.")
